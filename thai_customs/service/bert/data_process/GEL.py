import os
import json
import time
from logzero import logger
from sklearn.decomposition import PCA
from bert_serving.client import BertClient
import v_thai_customs.service

class GenerateEmbeddingLayer():

    def __init__(self, configs, name, clean_json):

        start_time = time.time()
        self.bc = BertClient()
        self.auxiliary_vector_inv = []
        self.auxiliary_vector_pl = []
        self.auxiliary_path = os.path.join(v_thai_customs.service.__file__[:-12], configs.auxiliary_path)
        self.get_auxiliary_vector()
        self.pca = PCA(n_components = 128)

        self.name = name[:name.find(".")]
        self.clean_json = clean_json
        self.vec_768_json = None
        self.vec_128_json = None

        logger.info("--------------- Bert Server Loaded  Time Consume:{}ms ---------------".format(
            '%.0f' % ((time.time() - start_time) * 1000)))

    def Sentence2Vec(self, sentence):
        return self.bc.encode(sentence)

    def get_auxiliary_vector(self):
        for inv_name in os.listdir(os.path.join(self.auxiliary_path, 'Invoice')):
            invfile = open(os.path.join(os.path.join(self.auxiliary_path, 'Invoice'), inv_name), encoding="utf-8")
            invlist = invfile.read()
            inv_str = json.loads(invlist, encoding="utf-8")
            invfile.close()
            for i in range(0, len(inv_str)):
                self.auxiliary_vector_inv.append(inv_str[i]["embedding"])

        for pl_name in os.listdir(os.path.join(self.auxiliary_path, 'PackingList')):
            plfile = open(os.path.join(os.path.join(self.auxiliary_path, 'PackingList'), pl_name), encoding="utf-8")
            pllist = plfile.read()
            pl_str = json.loads(pllist, encoding="utf-8")
            plfile.close()
            for i in range(0, len(pl_str)):
                self.auxiliary_vector_pl.append(pl_str[i]["embedding"])
