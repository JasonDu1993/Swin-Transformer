# -*- coding: utf-8 -*-
# @Time    : 2022/3/31 16:59
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : freeze_layer.py
# @Software: PyCharm
from collections import Iterable


def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, param in model.named_parameters():
        for ln in layer_names:
            if name.startswith(ln):
                param.requires_grad = not freeze
        # if name not in layer_names:
        #     continue
        # for param in child.parameters():
        #     # print(param.name)
        #     param.requires_grad = not freeze


def set_unfreeze_by_names(model, layer_names, unfreeze=True):
    """表示以layer_names里面的值开头的部分不冻结，其他层都冻结，适用于微调某一部分的情况

    Args:
        model:
        layer_names: list
        unfreeze:
    """
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, param in model.named_parameters():
        for ln in layer_names:
            if not name.startswith(ln):
                param.requires_grad = not unfreeze
            else:
                param.requires_grad = unfreeze


def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)


def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)


def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)


def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)


def set_freeze_all(model, freeze=True):
    for idx, child in enumerate(model.children()):
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_all(model):
    set_freeze_all(model, True)


def unfreeze_all(model):
    set_freeze_all(model, False)
