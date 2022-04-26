# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
from __future__ import division
import torch
import math
import random
import numpy as np
import torch
import torch.distributed as dist
from collections import defaultdict
import copy
from tqdm import tqdm

from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler, RandomSampler


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 replace=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.replace = replace
        self.unif_sampling_flag = False

    def __iter__(self):
        # deterministically shuffle based on epoch
        if not self.unif_sampling_flag:
            self.generate_new_list()
        else:
            self.unif_sampling_flag = False
        return iter(self.indices[self.rank * self.num_samples:(self.rank + 1) *
                                                              self.num_samples])

    def generate_new_list(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            if self.replace:
                indices = torch.randint(
                    low=0,
                    high=len(self.dataset),
                    size=(len(self.dataset),),
                    generator=g).tolist()
            else:
                indices = torch.randperm(
                    len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        self.indices = indices

    def set_uniform_indices(self, labels, num_classes):
        self.unif_sampling_flag = True
        assert self.shuffle, "Using uniform sampling, the indices must be shuffled."
        np.random.seed(self.epoch)
        assert (len(labels) == len(self.dataset))
        N = len(labels)
        size_per_label = int(N / num_classes) + 1
        indices = []
        images_lists = [[] for i in range(num_classes)]
        for i, l in enumerate(labels):
            images_lists[l].append(i)
        for i, l in enumerate(images_lists):
            if len(l) == 0:
                continue
            indices.extend(
                np.random.choice(
                    l, size_per_label, replace=(len(l) <= size_per_label)))
        indices = np.array(indices)
        np.random.shuffle(indices)
        indices = indices[:N].astype(np.int).tolist()

        # add extra samples to make it evenly divisible
        assert len(indices) <= self.total_size, \
            "{} vs {}".format(len(indices), self.total_size)
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size, \
            "{} vs {}".format(len(indices), self.total_size)
        self.indices = indices


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


class DistributedGivenIterationSampler(Sampler):

    def __init__(self,
                 dataset,
                 total_iter,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 last_iter=-1):
        rank, world_size = get_dist_info()
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter * self.batch_size

        self.indices = self.gen_new_list()

    def __iter__(self):
        return iter(self.indices[(self.last_iter + 1) * self.batch_size:])

    def set_uniform_indices(self, labels, num_classes):
        np.random.seed(0)
        assert (len(labels) == len(self.dataset))
        N = len(labels)
        size_per_label = int(N / num_classes) + 1
        indices = []
        images_lists = [[] for i in range(num_classes)]
        for i, l in enumerate(labels):
            images_lists[l].append(i)
        for i, l in enumerate(images_lists):
            if len(l) == 0:
                continue
            indices.extend(
                np.random.choice(
                    l, size_per_label, replace=(len(l) <= size_per_label)))
        indices = np.array(indices)
        np.random.shuffle(indices)
        indices = indices[:N].astype(np.int)
        # repeat
        all_size = self.total_size * self.world_size
        indices = indices[:all_size]
        num_repeat = (all_size - 1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]
        np.random.shuffle(indices)
        # slice
        beg = self.total_size * self.rank
        indices = indices[beg:beg + self.total_size]
        assert len(indices) == self.total_size
        # set
        self.indices = indices

    def gen_new_list(self):

        # each process shuffle all list with same seed, and pick one piece according to rank
        np.random.seed(0)

        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size - 1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)
        beg = self.total_size * self.rank
        indices = indices[beg:beg + self.total_size]

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        # return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size

    def set_epoch(self, epoch):
        pass


class _BatchMergeSampler(Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch

    Args:
        base_sampler_creation: function, input a Dataset, return a Sampler.
    """

    def __init__(self, dataset, dataset_batch_size, base_sampler_creation, world_size=1):
        super(_BatchMergeSampler, self).__init__(data_source=dataset)
        assert isinstance(dataset, torch.utils.data.ConcatDataset)
        self.dataset = dataset
        self.base_sampler_creation = base_sampler_creation

        self.dataset_batch_size = dataset_batch_size
        self.batch_size = sum(self.dataset_batch_size)
        self.number_of_datasets = len(dataset.datasets)

        self.dataset_iter_num = [math.ceil(len(self.dataset.datasets[i]) / (world_size * self.dataset_batch_size[i]))
                                 for i in
                                 range(self.number_of_datasets)]
        self.slice_id_points()

    def __len__(self):
        return max(self.dataset_iter_num) * self.batch_size

    def slice_id_points(self):
        self.id_marks = []
        acc = 0
        for bs in self.dataset_batch_size:
            self.id_marks.append((acc, acc + bs))
            acc += bs

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = self.base_sampler_creation(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size

        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        max_idx = np.argmax(self.dataset_iter_num)
        epoch_samples = self.dataset_iter_num[max_idx] * self.batch_size

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []

                samples_to_grab = self.dataset_batch_size[i]
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedBatchMergeSampler(_BatchMergeSampler):
    def __init__(self, dataset, dataset_batch_size, shuffle=True):
        assert dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size()
        super().__init__(dataset, dataset_batch_size, lambda d: _DistributedSampler(d, shuffle=shuffle),
                         world_size=world_size)


class BatchMergeSampler(_BatchMergeSampler):
    def __init__(self, dataset, dataset_batch_size):
        super().__init__(dataset, dataset_batch_size, lambda d: RandomSampler(d))


class DistributedGroupSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - dataset (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    # def __init__(self, dataset, batch_size, num_instances):
    def __init__(self, dataset, num_class, num_per_class, num_neg=0, num_replicas=None, rank=None, seed=0,
                 num_template=0):
        batch_size = num_class * num_per_class
        num_instances = num_per_class
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index in tqdm(range(len(self.dataset)), total=len(self.dataset), desc="Iter data source"):
            meta = self.dataset.get_source(index)
            pid = meta[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in tqdm(self.pids, total=len(self.pids), desc="Building sample index"):
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch
