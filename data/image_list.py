# -*- coding=UTF-8 -*-
import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class ImageListDataset(Dataset):

    def __init__(self, root, list_file, return_label=True, transform=None, offset=0):
        self.labels = []
        self.fns = []
        with open(list_file, "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                fn = line_sp[0]
                if return_label:
                    self.labels.append(int(line_sp[1]) + offset)
                self.fns.append(os.path.join(root, fn))
        self.num_class = len(set(self.labels))
        self.max_class = max(set(self.labels))
        self.num_imgs = len(self.fns)
        self.transform = transform
        self.return_label = return_label

    def __len__(self):
        return len(self.fns)

    def get_length(self):
        return len(self)

    def __getitem__(self, idx):
        img = Image.open(self.fns[idx].encode("utf-8"))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.return_label:
            target = self.labels[idx]
            return img, target
        else:
            return img

    def get_source(self, idx):
        output = (self.fns[idx],)
        if self.return_label:
            output = output + (self.labels[idx],)
        return output
