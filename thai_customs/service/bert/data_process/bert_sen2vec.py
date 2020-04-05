import time
import random
import numpy as np

from logzero import logger

class GenerateEmbeddingLayer():

    def __init__(self, clean_json):
        self.clean_json = clean_json
        self.vec_768_json = None
        self.vec_128_json = None

    def vec768_generation(self, configs):
        start_time = time.time()
        self.vec_768_json = self.clean_json

        item_rec = []
        for idx, item in enumerate(self.vec_768_json):
            if item["rec"]:
                item_rec.append(item["rec"])
            else:
                item_rec.append("default")
        embedding_vec = configs.GEL.Sentence2Vec(item_rec).tolist()

        for idx, item in enumerate(self.vec_768_json):
            if item["rec"]:
                self.vec_768_json[idx]["embedding"] = [embedding_vec[idx]]
            else:
                tempvec = [np.zeros([768]).tolist()]
                self.vec_768_json[idx]["embedding"] = tempvec

        logger.info('***** 768-dim Vector Generated: {}ms'.format('%.1f'%((time.time() - start_time)*1000)))

    def vec768_to_128(self, configs, img_type):
        start_time = time.time()
        flag = 0
        self.vec_128_json = self.vec_768_json

        for idx, item in enumerate(self.vec_128_json):
            tempvec = item["embedding"]
            if flag == 0:
                tempveclist = tempvec
                flag = 1
            else:
                tempveclist = np.concatenate((tempveclist, tempvec))

        rec_item_num = tempveclist.shape[0]
        if img_type == 'Invoice':
            total_pca_vec = 22000
            auxiliary_vector = configs.GEL.auxiliary_vector_inv
        else:
            total_pca_vec = 1700
            auxiliary_vector = configs.GEL.auxiliary_vector_pl

        supplement_num = total_pca_vec - rec_item_num
        if supplement_num > 0:
            supplement_vector = random.sample(auxiliary_vector, supplement_num)
            supplement_vector = np.array(supplement_vector)
            tempveclist = np.concatenate((tempveclist, np.squeeze(supplement_vector)))

        configs.GEL.pca.fit(tempveclist)
        tempveclist = configs.GEL.pca.transform(tempveclist)

        for idx, item in enumerate(self.vec_128_json):
            self.vec_128_json[idx]["embedding"] = tempveclist[idx].tolist()

        logger.info('***** 128-dim Vector Generated: {}ms'.format('%.2f'%((time.time() - start_time)*1000)))
        return self.vec_128_json