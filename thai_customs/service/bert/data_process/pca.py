import gensim
import numpy as np
import time
from logzero import logger
from sklearn.decomposition import PCA
import os
import cv2
import json


if __name__ == '__main__':
    folder_path = "/mnt/videt/data/document_analysis/data/layout/test/gt"
    image_folder = "/mnt/videt/data/document_analysis/data/layout/test/image"
    for json_name in os.listdir(folder_path):
        logger.info(json_name)
        image_path = os.path.join(image_folder,json_name.replace("json","png"))
        image = cv2.imread(image_path)
        jsonfile = open(os.path.join(folder_path,json_name), encoding="utf-8")
        jsonlist = jsonfile.read()
        tempjson = json.loads(jsonlist, encoding="utf-8")
        jsonfile.close()
        max_long = 512
        ratio = max_long/min(image.shape[0],image.shape[1])
        resize_image = cv2.resize(image,None,None,ratio,ratio,interpolation=cv2.INTER_CUBIC)
        X_train = []
        for idx,item in enumerate(tempjson):
            X_train.append(np.array(item["embedding"]))
            # logger.info(X_train.shape)
        pca = PCA(n_components = 128)
        X_train = np.array(X_train)
        logger.info(np.mean(X_train).shape)
        logger.info(X_train.shape)
        X_train = X_train - np.mean(X_train)
        X_fit = pca.fit_transform(X_train)
        for x in X_fit:
            logger.info(x.shape)
        U1 = pca.components_
        logger.info(X_fit[0])
        cv2.imshow('image',image)
        cv2.waitKey(0)
