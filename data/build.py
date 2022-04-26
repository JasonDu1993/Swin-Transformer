# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import math
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler
from data.image_list import ImageListDataset
from data.samplers import BatchMergeSampler, DistributedBatchMergeSampler

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=False
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS if not config.debug else 0,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS if not config.debug else 0,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_val_loader(config):
    config.defrost()
    dataset_val, config.MODEL.NUM_CLASSES = build_dataset(is_train=False, config=config)
    config.freeze()
    if dist.get_rank() == 0:
        print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=False
        )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS if not config.debug else 0,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )
    return dataset_val, [data_loader_val]


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            # root = os.path.join(config.DATA.DATA_PATH, prefix)
            # dataset = datasets.ImageFolder(root, transform=transform)
            # dataset = ImageListDataset(root, list_path, transform=transform, offset=0)
            index = -1
            for i, r in enumerate(config.DATA.VAL.SINGLE_DATA_ROOTS):
                if "ImageNet21k" in r:
                    index = i
            if is_train:
                root = config.DATA.DATA_ROOTS[index]
                list_path = config.DATA.DATA_PATHS[index]
            else:
                root = config.DATA.VAL.SINGLE_DATA_ROOTS[index]
                list_path = config.DATA.VAL.SINGLE_DATA_PATHS[index]
            if dist.get_rank() == 0:
                print("{} root:{}\n{} list:{}".format(prefix, root, prefix, list_path))
            dataset = ImageListDataset(root, list_path, transform=transform, offset=0)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet22K':
        index = -1
        for i, r in enumerate(config.DATA.VAL.SINGLE_DATA_ROOTS):
            if "ImageNet21k" in r:
                index = i
        if is_train:
            root = config.DATA.DATA_ROOTS[index]
            list_path = config.DATA.DATA_PATHS[index]
        else:
            root = config.DATA.VAL.SINGLE_DATA_ROOTS[index]
            list_path = config.DATA.VAL.SINGLE_DATA_PATHS[index]
        dataset = ImageListDataset(root, list_path, transform=transform, offset=0)
        nb_classes = 21841  # 对于官方训练的模型类别数是
        # raise NotImplementedError("Imagenet-22K will come soon.")
    elif config.DATA.DATASET.lower() == "cifar10":
        dataset = datasets.CIFAR10(root="/dataset/dataset/cifar10", train=is_train)
        nb_classes = 10
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def compute_batch_size(bs, size_list):
    """根据数据集的大小计算每个batch内该数据集的样本数

    Args:
        opt:

    Returns:

    """
    size_list = [s * 1. / sum(size_list) for s in size_list]
    batch_size = [max(2, math.ceil(s * bs)) for s in size_list]
    for i, bs in enumerate(batch_size):
        if bs % 2 != 0:
            batch_size[i] = bs + 1
    if sum(batch_size) % 2 != 0:
        batch_size[-1] = batch_size[-1] + 1
    # batch_size[-1] = train_bs - sum(batch_size[0: -1])
    if dist.get_rank() == 0:
        print("The final batch size is {}, total {}".format(batch_size, sum(batch_size)))
    return batch_size


def build_loader_for_multi_dataset(config):
    if dist.get_rank() == 0:
        print("build_loader_for_multi_dataset")
    config.defrost()
    datasets_trains = []
    datasets_num_classes = [0]
    img_nums = []
    transform = build_transform(True, config)
    for i, list_path in enumerate(config.DATA.DATA_PATHS):
        root = config.DATA.DATA_ROOTS[i]
        if dist.get_rank() == 0:
            print("train root:{}\ntrain list:{}".format(root, list_path))
        il = ImageListDataset(root, list_path, transform=transform, offset=datasets_num_classes[-1])
        datasets_trains.append(il)
        datasets_num_classes.append(il.max_class + 1)  # +1 是为了防止两个数据集连接时导致重复的类
        img_nums.append(il.num_imgs)

    # compute bs
    if "BATCH_SIZES" not in config.DATA:
        config.DATA.BATCH_SIZES = config.DATA.BATCH_SIZE

    if not isinstance(config.DATA.BATCH_SIZES, list):
        config.DATA.BATCH_SIZES = compute_batch_size(config.DATA.BATCH_SIZES, img_nums)
    dataset_train = ConcatDataset(datasets_trains)
    # config.MODEL.NUM_CLASSES = datasets_num_classes[-1]
    config.MODEL.NUM_CLASSES = datasets_num_classes
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        # sampler_train = torch.utils.data.DistributedSampler(
        #     dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        # )
        sampler_train = DistributedBatchMergeSampler(
            dataset_train, config.DATA.BATCH_SIZES
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=sum(config.DATA.BATCH_SIZES),
        num_workers=config.DATA.NUM_WORKERS if not config.debug else 0,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_train.datasets_num_classes = datasets_num_classes
    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        if isinstance(config.MODEL.NUM_CLASSES, list):
            mixup_fn = []
            for i in range(len(datasets_num_classes[1:])):
                n_cls = config.MODEL.NUM_CLASSES[i + 1] - config.MODEL.NUM_CLASSES[i]
                mixup_fn_i = Mixup(
                    mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX,
                    cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                    prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
                    label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=n_cls)
                mixup_fn.append(mixup_fn_i)
        else:
            mixup_fn = Mixup(
                mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
                label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    # 验证集
    transform_val = build_transform(False, config)
    datasets_vals = []
    data_loader_val = []
    for i, list_path in enumerate(config.DATA.VAL.DATA_PATHS):
        # classfier_index = config.DATA.VAL.CLASSFIER_INDEXS[i]
        root = config.DATA.VAL.DATA_ROOTS[i]
        if dist.get_rank() == 0:
            print("val root:{}\nval list:{}".format(root, list_path))
        il = ImageListDataset(root, list_path, transform=transform_val,
                              offset=0)
        datasets_vals.append(il)
        if config.TEST.SEQUENTIAL:
            sampler_val = torch.utils.data.SequentialSampler(il)
        else:
            sampler_val = torch.utils.data.distributed.DistributedSampler(
                il, shuffle=False
            )

        data_loader = torch.utils.data.DataLoader(
            il, sampler=sampler_val,
            batch_size=config.DATA.VAL.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS if not config.debug else 0,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False
        )
        data_loader_val.append(data_loader)
    dataset_val = ConcatDataset(datasets_vals)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn
