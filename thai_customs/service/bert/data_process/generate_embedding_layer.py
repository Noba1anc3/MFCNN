# import gensim
import numpy as np
import time
from logzero import logger
import os
import cv2
import json

def generate_segment_gt(image, boxes, label):
    class_map = np.zeros((image.shape[0],image.shape[1]),dtype=np.float)
    boxes = np.array(boxes)
    label = np.array(label)
    for i in range(1,3):
        idxs = np.where(label == i)
        cv2.fillPoly(class_map,boxes[idxs],i,1)
    return class_map

def triarea(a,b,c):
    return 0.5*(a[0]*b[1]+b[0]*c[:,1]+c[:,0]*a[1]-a[0]*c[:,1]-b[0]*a[1]-c[:,0]*b[1])

def within(points,b):
    vis = np.ones((len(points),1))
    for i in range(b.shape[0]):
        begin,end = b[i],b[(i+1)%b.shape[0]]
        area = triarea(begin,end,points)
        vis[np.where(area<0)] = 0
        if i%2==0:
            vis[np.where(np.abs(area)<1e-5)] = 0
    return vis

def generate_embedding_layer(image , boxes, embedding, segment_gt):
    embedding_feature = np.zeros((image.shape[0],image.shape[1],128),dtype=np.float)
    logger.info(embedding_feature.shape)
    pointx,pointy = np.where(segment_gt[:,:]>0)
    pointx = pointx.reshape((pointx.shape[0],1))
    pointy = pointy.reshape((pointy.shape[0],1))
    points = np.concatenate((pointy,pointx),1)
    show_map = np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
    logger.info(len(points))
    for idx,box in enumerate(boxes):
        box = np.array(box)
        # cv2.fillPoly(embedding_feature,box,embedding[idx],1)            
        postive_idx = within(points,box)
        postive_point = points[np.where(postive_idx>0)[0],:]
        postive_y = postive_point[:,0]
        postive_x = postive_point[:,1]
        sub_embedding = embedding[idx]
        logger.info(sub_embedding.shape)
        embedding_feature[postive_x,postive_y] = sub_embedding
    embedding_feature = np.array(embedding_feature)
    return embedding_feature


if __name__ == '__main__':
    folder_path = "/home/videt/Projects/document_analysis/data/layout/train/gt"
    image_folder = "/home/videt/Projects/document_analysis/data/layout/train/image"
    for json_name in os.listdir(folder_path):
        logger.info(json_name)
        image_path = os.path.join(image_folder,json_name.replace("json","png"))
        image = cv2.imread(image_path)
        jsonfile = open(os.path.join(folder_path,json_name), encoding="utf-8")
        jsonlist = jsonfile.read()
        tempjson = json.loads(jsonlist, encoding="utf-8")
        jsonfile.close()
        boxes = []
        recos = []
        label = []
        embedding = []
        for idx,item in enumerate(tempjson):
            boxes.append(item["det"])
            recos.append(item["rec"])
            label.append(item["cls"])
            embedding.append(item["embedding"][:128])
        boxes = np.array(boxes)
        label = np.array(label)
        embedding = np.array(embedding)
        segment_gt = generate_segment_gt(image,boxes,label)
        embedding_layer = generate_embedding_layer(image,boxes,embedding,segment_gt)
        logger.info(embedding_layer.shape)
        save_folder = "/mnt/videt/data/document_analysis/data/layout/train/embedding_gt"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        np.save(save_folder+'/'+json_name.replace("json","npy"),embedding_layer)
    # vec = s2v("Pracking list of Printed Circuit Board")
