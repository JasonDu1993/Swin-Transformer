# -*- coding: utf-8 -*-
# @Time    : 2022/4/13 19:18
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : viewer.py
# @Software: PyCharm
import os
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist


class Viewer(object):
    def __init__(self):
        self.step = 0
        self.writer = None

    def init_tensorboard(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def _check_tensorboard_(self):
        return self.writer is not None

    def set_step(self, step):
        self.step = step

    def add_scalar(self, name, value, step):
        if self._check_tensorboard_() and dist.get_rank() == 0:
            self.writer.add_scalar(name, value, step)


viewer = Viewer()
