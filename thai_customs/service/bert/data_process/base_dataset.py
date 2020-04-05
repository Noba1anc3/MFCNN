"""
@Description:
@version: V1.0
@Company: VIDET
@Author: WUSHUFAN
@Date: 2019-03-04 10:04:08
"""
# -*- coding: utf-8 -*-
# Copyright(c) 2018-present, Videt Tech. All rights reserved.
# @Project : MSR
# @Time    : 19-2-18 上午9:08
# @Author  : kongshuchen
# @FileName: dataproducer.py
# @Software: PyCharm
import numpy as np
import cv2
import torch.utils.data as data
import os
import random
import math
import torch
import torchvision.transforms as Transforms
import copy
from scipy import ndimage
from src.utils.misc import load_json, getImgPath, norm2


class CustomDataset(data.Dataset):

    def __init__(self, root_dir, phase='train', transform=None):
        super(CustomDataset, self).__init__()
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.imgs_path_list = self.get_imgs_path(root_dir, phase)
        self.labels_path_list = self.get_labels_path(root_dir, phase)

    def get_imgs_path(self, root_dir, phase):
        return os.listdir(os.path.join(root_dir, phase, 'image'))

    def get_labels_path(self, root_dir, phase):
        return os.listdir(os.path.join(root_dir, phase, 'json'))

    def __getitem__(self, index):
            img_path = self.imgs_path_list[index]
            img = cv2.imread(img_path)
            if self.phase == 'val':
                if self.transform:
                    img, _ = self.transform(img)
                img = img.transpose(2, 0, 1)
                return img

            boxes = self.get_gt()

