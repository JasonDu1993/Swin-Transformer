# -*- coding: utf-8 -*-
# @Time    : 2022/4/11 11:11
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : check_image.py
# @Software: PyCharm
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SequentialSampler
from data.image_list import ImageListDataset
batch_size = 2
root= "/zhouyafei/image_recog_data/ImageNet21k/"
list_file = "/zhouyafei/image_recog_data/ImageNet21k/labels/train.txt"
dataset = ImageListDataset(root, list_file)
num_workers = 2
# shuffle为True时，sampler 应该为None
# batch_sampler选项与batch_size(不等于1时会报错,为1或者None不会报错), shuffle, sampler和drop_last互斥
sampler = SequentialSampler(dataset)
rand_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                         num_workers=num_workers)
for data in rand_loader:
    print(data)