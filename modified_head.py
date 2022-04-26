# -*- coding: utf-8 -*-
# @Time    : 2022/4/22 17:53
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : modified_head.py
# @Software: PyCharm
import torch
from models import build_model
from main import parse_option
from data.build import build_transform
from data.image_list import ImageListDataset
import torch.utils.data
from utils import load_checkpoint

if __name__ == '__main__':
    config = parse_option()
    bs = 4
    classfier_index = -1
    model = build_model(config)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    lr_scheduler = None  # 测试模型不需要这个
    optimizer = None  # 测试模型不需要这个
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    del checkpoint
    torch.cuda.empty_cache()
    print(msg)
    model.eval()
    root = "/zhouyafei/image_recog_data/ImageNet/ILSVRC2015/Data/CLS-LOC"
    list_path = "/zhouyafei/image_recog_data/ImageNet/labels/train1k_20.txt"
    transform = build_transform(False, config)
    dataset = ImageListDataset(root, list_path, transform=transform, offset=0)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=bs,
        num_workers=config.DATA.NUM_WORKERS if not config.debug else 0,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images, mode="val", head_index=classfier_index)

        # if config.AMP_OPT_LEVEL == "O2":
        #     output = output.half()
        # output = head(output)
        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
