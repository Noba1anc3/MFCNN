#!/usr/bin/env python
# coding=UTF-8

import numpy as np
import cv2
import torch
import copy
import warnings

warnings.filterwarnings("ignore")
from configs import cfg
from datasets.data_preprocessing import TrainAugmentation

class InferenceDataset():

    def __init__(self, image, gt, class_num):
        self.image = image
        self.gt = gt
        self.class_num = class_num
        self.transform = TrainAugmentation(cfg)

    def get_inference_data(self):
        image = self.image
        boxes, recos, label, embedding = self.load_data()

        image, boxes = self.transform(image, copy.deepcopy(boxes))


        segment_gt = self.generate_segment_gt(image, boxes, label)
        embedding_layer = self.generate_embedding_layer(image, boxes, embedding, segment_gt, 128)
        image = image.transpose(2, 0, 1)
        embedding_layer = embedding_layer.transpose(2, 0, 1)

        image = image[np.newaxis, ...]
        segment_gt = segment_gt[np.newaxis, ...]
        embedding_layer = embedding_layer[np.newaxis, ...]
        boxes = boxes[np.newaxis, ...]
        label = label[np.newaxis, ...]

        image = torch.from_numpy(image)
        segment_gt = torch.from_numpy(segment_gt)
        embedding_layer = torch.from_numpy(embedding_layer)
        boxes = torch.from_numpy(boxes)
        label = torch.from_numpy(label)
        return image, segment_gt, embedding_layer, boxes, label

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

    def load_data(self):
        gt_json = self.gt
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
        return boxes,recos,label,embedding
    
    def generate_segment_gt(self, image, boxes, label):
        class_map = np.zeros((image.shape[0],image.shape[1]),dtype=np.float)
        boxes = np.array(boxes)
        label = np.array(label)
        for i in range(1,self.class_num+1):
            idxs = np.where(label == i)
            cv2.fillPoly(class_map,boxes[idxs],i,1)

        return class_map


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
                           [139, 58, 58], [255, 193, 193], [139, 101, 8], [255, 185, 15], [139, 139, 0]
                           ])
    colorgt = colortable[img]
    colorgt = np.array(colorgt, dtype=np.uint8)

    return colorgt


