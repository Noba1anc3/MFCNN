
import numpy as np
from sklearn.decomposition import PCA
import os
import json
from bert_serving.client import BertClient
import random
import time
from logzero import logger


class GenerateEmbeddingLayer():

    def __init__(self, cfg, data_dir, bert_server_ip = "locahost"):
        self.data_dir = data_dir
        assert os.path.exists(self.data_dir)
        self.get_auxiliary_vector()
        self.pca = PCA(n_components=128)
        self.bc = BertClient(ip=bert_server_ip)
        self.cfg = cfg

    def Sentence2Vec(self, sentence):
        vec = self.bc.encode(sentence)
        return vec

    def Pre_vector(self, json_name):
        pre_gt_path = os.path.join(self.data_dir, "pre_gt", json_name)
        isExists = os.path.exists(pre_gt_path)
        if not isExists:
            return -1
        else:
            jsonfile = open(pre_gt_path)
            jsonlist = jsonfile.read()
            tempjson = json.loads(jsonlist)
            jsonfile.close()
            pre_vector = {}
            for i in range(0, len(tempjson)):
                pre_vector[tempjson[i]["rec"]] = tempjson[i]["embedding"]
            return pre_vector


    def generate_vector768(self, json):
        self.json = json
        tempjson = self.json
        item_rec = []
        for idx, item in enumerate(tempjson):
            if item["rec"]:
                item_rec.append(item["rec"])
            else:
                item_rec.append("default")
        start = time.time()
        embedding_vec = self.Sentence2Vec(item_rec).tolist()
        # logger.info(time.time()-start)
        for idx, item in enumerate(tempjson):
            if item["rec"]:
                tempjson[idx]["embedding"] = [embedding_vec[idx]]
            else:
                tempvec = [np.zeros([768]).tolist()]
                tempjson[idx]["embedding"] = tempvec
        self.json768 = tempjson

    def get_auxiliary_vector(self):
        folder_path = os.path.join(self.data_dir, "auxiliary_gt")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.auxiliary_vector = []
        for json_name in os.listdir(folder_path):
            jsonfile = open(os.path.join(folder_path, json_name), encoding="utf-8")
            jsonlist = jsonfile.read()
            tempjson = json.loads(jsonlist, encoding="utf-8")
            jsonfile.close()
            for i in range(0, len(tempjson)):
                self.auxiliary_vector.append(tempjson[i]["embedding"])

    def vec768to128(self):
        flag = 0
        tempjson = self.json768
        for idx, item in enumerate(tempjson):
            tempvec = item["embedding"]
            if flag == 0:
                tempveclist = tempvec
                flag = 1
            else:
                tempveclist = np.concatenate((tempveclist, tempvec))
        
        rec_item_num = tempveclist.shape[0]

        total_pca_vec = 5000
        supplement_num = total_pca_vec - rec_item_num

        if supplement_num > 0:
            supplement_vector = random.sample(self.auxiliary_vector, supplement_num)
            supplement_vector = np.array(supplement_vector)
            tempveclist = np.concatenate((tempveclist, np.squeeze(supplement_vector)))

        self.pca.fit(tempveclist)
        tempveclist = self.pca.transform(tempveclist)

        for idx, item in enumerate(tempjson):
            tempjson[idx]["embedding"] = tempveclist[idx].tolist()

        return tempjson


