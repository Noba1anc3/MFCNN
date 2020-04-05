#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@version: 
@Company: VIDET
@Author: ZHOUZHAO
@LastEditors: ZHOUZHAO
@Date: 2019-03-11 18:53:56
@LastEditTime: 2019-03-15 18:06:24
'''

import numpy as np
import cv2
import torch.utils.data as data
import os
import random
import math
import json
import torch
import torchvision.transforms as Transforms
import copy
import traceback
# from src.transforms.transforms import Resize,Scale
from logzero import logger
import time
import warnings
from configs import config
warnings.filterwarnings("ignore")
from src.config import cfg
import argparse


# 海关数据集.png, 通用发票数据集.jpg

class LayoutDataset(data.Dataset):

    def __init__(self, cfg, args, phase='train', transform=None):
        super(LayoutDataset, self).__init__()
        self.root_dir = cfg.DATASET.ROOT_DIR
        self.phase = phase
        self.class_num = cfg.MODEL.NUM_CLASSES-1
        self.image_dir = os.path.join(self.root_dir, 'image')
        self.gt_dir = os.path.join(self.root_dir, 'gt128')
        self.embedding_gt_dir = os.path.join(self.root_dir, 'embedding_gt')
        self.init_data()
        self.transform = transform
        self.embedding_dim = args.embedding_dim

    def init_data(self):
        image_name_list = os.listdir(self.image_dir)
        self.image_name_list = image_name_list
        self.image_path_list = [os.path.join(
            self.image_dir, image_name) for image_name in image_name_list]

        self.gt_path_list = [os.path.join(self.gt_dir, image_name.replace(cfg.DATASET.image_format, '.json')) for image_name
                             in image_name_list]
        self.embedding_gt_path_list = [os.path.join(self.embedding_gt_dir, image_name.replace(cfg.DATASET.image_format, '.npy')) for
                                       image_name
                                       in image_name_list]

    def __getitem__(self, index):
        try:
            image,boxes,recos,label,embedding = self.load_data(index)
            embedding_dim = self.embedding_dim
            image_name = self.image_name_list[index]

            if self.phase == 'val':
                if self.transform:
                    image, _ = self.transform(image)
                image = image.transpose(2, 0, 1)
                return image_name,image,boxes,label
            if self.transform:
                image, boxes = self.transform(image, copy.deepcopy(boxes))
            segment_gt = self.generate_segment_gt(image,boxes,label)
            embedding_layer = self.generate_embedding_layer(image,boxes,embedding,segment_gt,embedding_dim)
            image = image.transpose(2, 0, 1)
            embedding_layer = embedding_layer.transpose(2, 0, 1)

            # image = np.array(image,dtype=np.float32)
            return image_name,image,segment_gt,embedding_layer,boxes,label
        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.debug("image_path:"+self.image_path_list[index])
            logger.info('dataloader error')
            logger.info("image_path:".format(self.image_path_list[index]))
            return "",np.array([]),np.array([]),np.array([]),np.array([])
    
    def triarea(self,a,b,c):
        return 0.5*(a[0]*b[1]+b[0]*c[:,1]+c[:,0]*a[1]-a[0]*c[:,1]-b[0]*a[1]-c[:,0]*b[1])

    def within(self,points,b):
        vis = np.ones((len(points),1))
        for i in range(b.shape[0]):
            begin,end = b[i],b[(i+1)%b.shape[0]]
            area = self.triarea(begin,end,points)
            vis[np.where(area<0)] = 0
            if i%2==0:
                vis[np.where(np.abs(area)<1e-5)] = 0
        return vis

    def generate_embedding_layer(self, image , boxes, embedding, segment_gt, embedding_dim):
        embedding_feature = np.zeros((image.shape[0],image.shape[1],embedding_dim),dtype=np.float)
        pointx,pointy = np.where(segment_gt[:,:]>0)
        pointx = pointx.reshape((pointx.shape[0],1))
        pointy = pointy.reshape((pointy.shape[0],1))
        points = np.concatenate((pointy,pointx),1)
        show_map = np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
        for idx,box in enumerate(boxes):
            box = np.array(box)
            # cv2.fillPoly(embedding_feature,box,embedding[idx],1)            
            postive_idx = self.within(points,box)
            postive_point = points[np.where(postive_idx>0)[0],:]
            postive_y = postive_point[:,0]
            postive_x = postive_point[:,1]
            sub_embedding = embedding[idx]
            sub_embedding = np.array(sub_embedding).reshape((embedding_dim))
            embedding_feature[postive_x,postive_y] = sub_embedding   # [:128]
        embedding_feature = np.array(embedding_feature)
        return embedding_feature

    def load_data(self, index):
        image_path = self.image_path_list[index]
        image = cv2.imread(image_path)
        if self.phase == "val":
            return image, [], [], []  # None, None, None
        gt_json_path = self.gt_path_list[index]
        jsonfile = open(gt_json_path, encoding="utf-8")
        jsonlist = jsonfile.read()
        gt_json = json.loads(jsonlist, encoding="utf-8")
        jsonfile.close()
        boxes = []
        recos = []
        label = []
        embedding = []
        for item in gt_json:
            boxes.append(item["det"])
            recos.append(item["rec"])
            label.append(item["cls"])
            embedding.append(item["embedding"])
        boxes = np.array(boxes)
        label = np.array(label)
        embedding = np.array(embedding)
        return image,boxes,recos,label,embedding

    def __len__(self):
        if "image_path_list" in dir(self):
            return len(self.image_path_list)
        else:
            return 0
    
    def generate_segment_gt(self, image, boxes, label):
        class_map = np.zeros((image.shape[0],image.shape[1]),dtype=np.float)
        boxes = np.array(boxes)
        label = np.array(label)
        for i in range(1,self.class_num+1):
            idxs = np.where(label == i)
            cv2.fillPoly(class_map,boxes[idxs],i,1)

        return class_map


def create_dataset(cfg, args, transfrom, phase):
    return LayoutDataset(cfg, args, phase=phase, transform=transfrom)


def colorimg(img):
    img = np.array(img)
    img = img.astype(int)
    colortable = np.array([[255, 255, 255], [34, 34, 178], [222, 173, 255], [191, 255, 0], [140, 0, 255],
                           [193, 37, 255], [205, 50, 50], [139, 87, 46], [90, 205, 106], [125, 107, 139],
                           [132, 112, 255], [173, 216, 230], [85, 107, 47], [0, 255, 127], [255, 255, 0]])
    colorgt = colortable[img]
    colorgt = np.array(colorgt, dtype=np.uint8)

    return colorgt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='./output504/net_iteration_002000.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument(
        "--config_file",
        default="./configs/config.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--with_embedding", default=True)
    parser.add_argument("--embedding_dim", type=int, default=128)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    trainset = LayoutDataset(
        cfg, args, transform=None, phase='train')
    train_dataloader = data.DataLoader(
        dataset=trainset, batch_size=1, shuffle=False, num_workers=1)
    n = 0
    for idx, (image_name,image,segment_gt,embedding_layer,boxes, label) in enumerate(train_dataloader):
        print(n)
        n += 1
        logger.info(image_name)
        image = image.numpy()[0,::]
        image = np.array(image,dtype=np.uint8)
        segment_gt = segment_gt.numpy()[0,::]
        segment_gt = colorimg(segment_gt)
        # segment_gt = np.array(segment_gt.numpy()[0,::],dtype=np.uint8)*28

        image = image.transpose(1,2,0)
        cv2.imshow('image',image)
        cv2.imshow('segment_gt', segment_gt)
        logger.info(embedding_layer.size())
        # logger.info(segment_gt.size())
        cv2.waitKey(0)
