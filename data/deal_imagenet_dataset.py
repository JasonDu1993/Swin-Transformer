# -*- coding: utf-8 -*-
# @Time    : 2022/4/6 11:31
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : deal_imagenet_dataset.py
# @Software: PyCharm
import os
import cv2
from time import time
import pandas as pd
import numpy as np
from collections import OrderedDict, defaultdict
import math


def unzip():
    root_dir = "/zhouyafei/image_recog_data/ImageNet"
    dst_dir = "/zhouyafei/image_recog_data/ImageNet21k"

    for i, name in enumerate(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if path.endswith(".zip"):
            dst_dir_images = os.path.join(dst_dir, "images", ".".join(name.split(".")[:-1]))
            if not os.path.exists(dst_dir_images):
                os.makedirs(dst_dir_images)
                cmd = "unzip -q -d " + dst_dir_images + " " + path
                print(cmd)
                os.system(cmd)


def del_some_file():
    root_dir = "/zhouyafei/image_recog_data/ImageNet"
    dst_dir = "/zhouyafei/image_recog_data/ImageNet21k"

    cnt = 0
    for i, name in enumerate(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if path.endswith(".zip"):
            dst_dir_images = os.path.join(dst_dir, "images", ".".join(name.split(".")[:-1]))
            if os.path.exists(dst_dir_images):
                print(path)
                os.remove(path)
                cnt += 1
    print("del {} file".format(cnt))


def get_dir_len(path):
    if path.endswith("/"):
        l = len(path)
    else:
        l = len(path) + 1
    return l


def gen_label():
    root_dir = "/zhouyafei/image_recog_data/ImageNet21k"
    src_dir = "/zhouyafei/image_recog_data/ImageNet21k/train"
    label_path = "/zhouyafei/image_recog_data/ImageNet21k/labels/train.txt"
    indexid_label = "/zhouyafei/image_recog_data/ImageNet21k/labels/class-descriptions.txt"

    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    os.makedirs(os.path.dirname(indexid_label), exist_ok=True)
    t0 = time()
    with open(label_path, "w") as fw1:
        with open(indexid_label, "w") as fw2:
            for i, name in enumerate(sorted(os.listdir(src_dir))):
                if (i + 1) % 1000 == 0:
                    print("spend {} s in {} class".format(time() - t0, i + 1))
                fw2.write(str(i) + " " + name + "\n")
                path_dir = os.path.join(src_dir, name)
                for j, p in enumerate(sorted(os.listdir(path_dir))):
                    path = os.path.join(path_dir, p)
                    fw1.write(path[get_dir_len(root_dir):] + " " + str(i) + "\n")
                    # try:
                    #     img = cv2.imread(path)
                    #     if img is None:
                    #         print(path)
                    #     else:
                    #         fw1.write(path[get_dir_len(root_dir):] + " " + str(i) + "\n")
                    # except Exception as e:
                    #     print(path)
                fw1.flush()
                fw2.flush()


def valid_img():
    root_dir = "/zhouyafei/image_recog_data/ImageNet21k"
    label_path = "/zhouyafei/image_recog_data/ImageNet21k/labels/train.txt"
    log_path = "/zhouyafei/image_recog_data/ImageNet21k/labels/valid_train_log.txt"

    t0 = time()
    file_set = set()
    if os.path.exists(log_path):
        with open(log_path, "r") as fr:
            for line in fr.readlines():
                file_set.add(line)
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(label_path, "r") as fr:
        with open(log_path, "a+") as fw:
            for i, line in enumerate(fr.readlines()):
                if line in file_set:
                    continue
                if (i + 1) % 1000 == 0:
                    print("valid spend {} s in {} images".format(time() - t0, i + 1))
                img_path = os.path.join(root_dir, line.strip().split(" ")[0])
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print("img is none", img_path)
                    else:
                        fw.write(line)
                except Exception as e:
                    print(img_path)
                # raise Exception


def gen_val_label():
    # root_dir = "/zhouyafei/image_recog_data/ImageNet21k/"
    # src_dir = "/zhouyafei/image_recog_data/ImageNet21k//val"
    root_dir = "/zhouyafei/image_recog_data/ImageNet/ILSVRC2015/Data/CLS-LOC/"
    src_dir = "/zhouyafei/image_recog_data/ImageNet/ILSVRC2015/Data/CLS-LOC/val"
    indexid_label = "/zhouyafei/image_recog_data/ImageNet21k/labels/class-descriptions.txt"

    label_path = "/zhouyafei/image_recog_data/ImageNet21k/labels/val.v2.txt"

    total_labels = {}
    with open(indexid_label, "r") as fr:
        for line in fr.readlines():
            index, label = line.strip().split(" ")
            total_labels[label] = index

    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as fw1:
        for i, name in enumerate(sorted(os.listdir(src_dir))):
            if name in total_labels:
                index = total_labels[name]
                # print(name, index)
                path_dir = os.path.join(src_dir, name)
                for j, p in enumerate(sorted(os.listdir(path_dir))):
                    path = os.path.join(path_dir, p)
                    fw1.write(path[get_dir_len(root_dir):] + " " + str(index) + "\n")
            else:
                print(name)


def find_empty_class():
    src_dir = "/zhouyafei/image_recog_data/ImageNet21k/train"
    # 11143 n04399382
    # 17491 n11653904
    for j, p in enumerate(sorted(os.listdir(src_dir))):
        path = os.path.join(src_dir, p)
        a = os.listdir(path)
        if len(a) == 0:
            print(j, p)


def new_val_txt():
    """主要是swin官方训练的模型类别总数只有21841，原因是训练集中不包含11143 n04399382(从0开始编号)，但是测试集中又有这个类，因此需要重新进行映射,，大于11143的label都需要减去1

    Returns:

    """
    path = "/zhouyafei/image_recog_data/ImageNet21k/labels/val.txt"
    dst_path = "./labels/val.swin.v2.txt"
    with open(path, "r") as fr:
        with open(dst_path, "w") as fw:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                img_path = line_sp[0]
                label = int(line_sp[1])
                if label == 11143:
                    label = 5564
                    # continue
                elif 9211 <= label <= 11136 or 15731 <= label <= 21816:
                    label = label + 1
                new_line = img_path + " " + str(label) + "\n"
                fw.write(new_line)


def check_val_label():
    path = "labels/val.txt"
    l1 = pd.read_csv(path, header=None, sep=" ")
    ll1 = l1.drop_duplicates(subset=[1], keep='first', inplace=False)
    path2 = "map22kto1k.txt"
    l2 = pd.read_csv(path2, header=None, sep=" ")
    a1 = np.array(list(ll1[1].tolist()))
    a2 = np.array(list(l2[0].tolist()))
    c = a1 != a2
    print(a1[c])
    print(a2[c])
    print(c.sum())
    d = a1 == a2
    e = a2[d].tolist()
    save_path = "labels/val.same.txt"
    with open(save_path, "w") as fw:
        with open(path, "r") as fr:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                img_path = line_sp[0]
                label = int(line_sp[1])
                if label > 9204:
                    continue
                if label in e:
                    fw.write(line)


def find_num():
    path = "labels/val.same.txt"
    l1 = pd.read_csv(path, header=None, sep=" ")
    ll1 = l1.drop_duplicates(subset=[1], keep='first', inplace=False)
    a = ll1[1].tolist()
    print(a)
    print(len(a))


def gen_val_1k_label():
    path = "labels/val.txt"
    src_path = "labels/val_map.txt"
    dst_path = "labels/val1k.txt"
    maps = {}
    with open(src_path, "r") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            label = line_sp[1]
            img_name = img_path.split("/")[-1]
            maps[img_name] = label
    with open(path, "r") as fr:
        with open(dst_path, "w") as fw:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                img_path = line_sp[0]
                img_name = img_path.split("/")[-1]
                label = maps[img_name]
                new_line = img_path + " " + label + "\n"
                fw.write(new_line)


def gen_imagenet1k_label():
    root_dir = "/zhouyafei/image_recog_data/ImageNet/ILSVRC2015/Data/CLS-LOC"
    src_dir = "/zhouyafei/image_recog_data/ImageNet/ILSVRC2015/Data/CLS-LOC/train"
    label_path = "/zhouyafei/image_recog_data/ImageNet/labels/train1k.txt"
    indexid_label = "/zhouyafei/image_recog_data/ImageNet/labels/class-descriptions-1k.txt"

    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    os.makedirs(os.path.dirname(indexid_label), exist_ok=True)
    t0 = time()
    with open(label_path, "w") as fw1:
        with open(indexid_label, "w") as fw2:
            for i, name in enumerate(sorted(os.listdir(src_dir))):
                if (i + 1) % 1000 == 0:
                    print("spend {} s in {} class".format(time() - t0, i + 1))
                fw2.write(str(i) + " " + name + "\n")
                path_dir = os.path.join(src_dir, name)
                for j, p in enumerate(sorted(os.listdir(path_dir))):
                    path = os.path.join(path_dir, p)
                    new_line = path[get_dir_len(root_dir):] + " " + str(i) + "\n"
                    fw1.write(new_line)
                    # try:
                    #     img = cv2.imread(path)
                    #     if img is None:
                    #         print(path)
                    #     else:
                    #         fw1.write(path[get_dir_len(root_dir):] + " " + str(i) + "\n")
                    # except Exception as e:
                    #     print(path)
                fw1.flush()
                fw2.flush()


def gen_val_desc():
    path = "labels/val1k.txt"
    save_path = "labels/class-descriptions-1k-val.txt"
    maps = OrderedDict()
    with open(path, "r") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            cls = img_path.split("/")[-2]
            label = line_sp[1]
            maps[label] = cls

    with open(save_path, "w") as fw:
        for key in maps.keys():
            new_line = key + " " + maps[key] + "\n"
            fw.write(new_line)


def select_some_sample(num_per_class=20):
    src_path = "/zhouyafei/image_recog_data/ImageNet/labels/train1k.txt"
    dst_path = "/zhouyafei/image_recog_data/ImageNet/labels/train1k_20.txt"
    select_label_num_dict = defaultdict(int)
    with open(src_path, "r", encoding='UTF-8') as fr:
        with open(dst_path, "w", encoding='UTF-8') as fw:
            for l in fr.readlines():
                src, label = l.strip().split(" ")
                cnt = select_label_num_dict.get(label, 0)
                cnt += 1
                select_label_num_dict[label] = cnt
                if cnt > num_per_class:
                    continue
                fw.write(l)


if __name__ == '__main__':
    # unzip()
    # del_some_file()
    # gen_label()
    # valid_img()
    # gen_val_label()
    # find_empty_class()
    # new_val_txt()
    # check_val_label()
    # gen_val_1k_label()
    # find_num()
    # gen_imagenet1k_label()
    # gen_val_desc()
    select_some_sample()
