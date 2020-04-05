#查看图片的ground truth
#查看图片的ground truth
#查看图片的ground truth
#查看图片的ground truth
#查看图片的ground truth
#查看图片的ground truth

import random
import time
import math
import torch

# from src.transforms.transforms import Resize,Scale
import os
import cv2
import sys
import json
import copy
import warnings
import traceback
sys.path.append('./')

import numpy as np
import torch.utils.data as data
from logzero import logger
from datasets.key_list import get_keylist
warnings.filterwarnings("ignore")

class LayoutDataset(data.Dataset):

    def __init__(self, root_dir, class_num, embedding_dim,  max_h=None, max_w=None, phase=[], transform=None):
        super(LayoutDataset, self).__init__()
        self.root_dir = root_dir
        self.phase = phase
        self.class_num = class_num
        # self.image_dir = os.path.join(root_dir, phase, 'image')
        # self.gt_dir = os.path.join(root_dir, phase, 'gt')
        # self.embedding_gt_dir = os.path.join(root_dir, phase, 'embedding_gt')
        self.init_data()
        self.transform = transform
        self.embedding_dim = embedding_dim

    def init_data(self):
        # image_name_list = os.listdir(self.image_dir)
        # self.image_name_list = image_name_list
        # self.image_path_list = [os.path.join(
        #     self.image_dir, image_name) for image_name in image_name_list]

        # 定额发票、翰航 .jpg
        # 海关 .png
        self.image_path_list = []
        self.gt_path_list = []
        self.image_name_list = []
        for image_set in self.phase:
            image_dir = os.path.join(self.root_dir, image_set, 'image')
            print(self.root_dir, image_set, image_dir)
            image_name_list = os.listdir(image_dir)
            gt_dir = os.path.join(self.root_dir, image_set, 'gt')
            image_path = [os.path.join(image_dir, image_name) for image_name in image_name_list]
            gt_path = [os.path.join(gt_dir, image_name.replace('png', 'json')) for image_name
                             in image_name_list]

            for sig_img_name in image_name_list:
                self.image_name_list.append(sig_img_name)
            for sig_img_path in image_path:
                self.image_path_list.append(sig_img_path)
            for sig_gt_path in gt_path:
                self.gt_path_list.append(sig_gt_path)

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
            if i%2 == 0:
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
    keylist = get_keylist(cfg)
    return LayoutDataset(root_dir=cfg.DATASET.ROOT_DIR, class_num=len(keylist), embedding_dim=args.embedding_dim, phase=phase, transform=transfrom, max_h=cfg.DATASET.MAX_H, max_w=cfg.DATASET.MAX_W)

def colorimg(img):
    img = np.array(img)
    img = img.astype(int)
    colortable = np.array([[255, 255, 255], [143, 250, 0], [222, 173, 255], [191, 255, 0], [140, 0, 255],
                           [193, 37, 255], [205, 50, 50], [139, 87, 46], [90, 205, 106], [125, 107, 139],
                           [132, 112, 255], [173, 216, 230], [85, 107, 47], [0, 255, 127], [255, 255, 0],
                           [122, 139, 139], [209, 238, 238], [154, 192, 205], [108, 166, 205], [0, 0, 238],
                           [93, 71, 139], [85, 26, 139], [224, 102, 255], [255, 187, 255], [139, 34, 82],
                           [139, 125, 123], [255, 228, 225], [193, 205, 193], [205, 179, 139], [147, 112, 219],
                           [255, 69, 0], [139, 54, 38], [238, 118, 0], [238, 149, 114], [139, 126, 102],
                           [139, 58, 58], [255, 193, 193], [139, 101, 8], [255, 185, 15], [139, 139, 0],

                           [255, 255, 255], [143, 250, 0], [222, 173, 255], [191, 255, 0], [140, 0, 255],
                           [193, 37, 255], [205, 50, 50], [139, 87, 46], [90, 205, 106], [125, 107, 139],
                           [132, 112, 255], [173, 216, 230], [85, 107, 47], [0, 255, 127], [255, 255, 0],
                           [122, 139, 139], [209, 238, 238], [154, 192, 205], [108, 166, 205], [0, 0, 238],
                           [93, 71, 139], [85, 26, 139], [224, 102, 255], [255, 187, 255], [139, 34, 82],
                           [139, 125, 123], [255, 228, 225], [193, 205, 193], [205, 179, 139], [147, 112, 219],
                           [255, 69, 0], [139, 54, 38], [238, 118, 0], [238, 149, 114], [139, 126, 102],
                           [139, 58, 58], [255, 193, 193], [139, 101, 8], [255, 185, 15], [139, 139, 0]
                           ])
    colorgt = colortable[img]
    colorgt = np.array(colorgt, dtype=np.uint8)

    return colorgt


if __name__ == '__main__':
    trainset = LayoutDataset(
        root_dir='./data/SIEMENS_PackingList/', class_num = 21, embedding_dim = 128, phase = ['train'], transform = None)
    train_dataloader = data.DataLoader(
        dataset = trainset, batch_size = 1, shuffle = False, num_workers = 1)

    for idx, (image_name,image, segment_gt, embedding_layer, boxes, label) in enumerate(train_dataloader):
        logger.info(image_name)
        image = image.numpy()[0,::]
        image = np.array(image, dtype = np.uint8)
        segment_gt = segment_gt.numpy()[0,::]
        segment_gt = colorimg(segment_gt)

        image = image.transpose(1,2,0)
        cv2.imshow('Image',image)
        cv2.imshow('Segmentation_GroundTruth', segment_gt)
        #logger.info(embedding_layer.size())
        cv2.waitKey(0)
