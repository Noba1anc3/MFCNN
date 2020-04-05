
import gensim
from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords as Stopwords
import numpy as np
import time
from sklearn.decomposition import pca
from logzero import logger
from sklearn.preprocessing import StandardScaler
import os
import json
from sklearn.decomposition import PCA


class Sentence2Vec(object):
    def __init__(self, word2vec_path):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            word2vec_path, binary=True)                  # encoding='ISO-8859-1', , unicode_errors='ignore'
        logger.info('loaded!')
        self.stopwords = filtered = Stopwords.words('english')

    def __call__(self, sentence):
        # tokenize and filter stopwords
        token_list = text_to_word_sequence(sentence)
        token_filtered = [w.lower()
                          for w in token_list if w.lower() not in self.stopwords]
        token_set = list(set(token_filtered))
        # extract token vectors from word2vec
        token_vecs = []
        for token in token_set:
            try:
                token_vecs.append(self.model[token])
            except KeyError:
                # print("token {} not in vocabulary".format(token))
                pass
        # 计算单词embedding的向量的平均值作为句子的向量
        vec_rs = []
        length = len(token_vecs)
        if length != 0:
            vec_rs = np.mean(np.array(token_vecs), axis=0)
        else:
            vec_rs = np.zeros((1, 300))
        return vec_rs

# 不经过pca
# 定额发票数据集59行 embedding = 0
# if __name__ == '__main__':
#     logger.info('loading GoogleNews-vectors-negative300.bin')
#     s2v = Sentence2Vec("./embedding/GoogleNews-vectors-negative300.bin")
#
#     folder_path = "/home/videt/Projects/document_analysis/data/defp/temp/gt"
#     for json_name in os.listdir(folder_path):
#         logger.info(json_name)
#         jsonfile = open(os.path.join(folder_path,json_name), encoding="utf-8")
#         jsonlist = jsonfile.read()
#         tempjson = json.loads(jsonlist, encoding="utf-8")
#         jsonfile.close()
#         for idx,item in enumerate(tempjson):
#             logger.info(idx)
#             # tempjson[idx]["embedding"] = s2v(item["rec"]).tolist()
#             tempjson[idx]["embedding"] = np.zeros([300]).tolist()
#         jsonfile = open(os.path.join(folder_path,json_name), "w")
#         jsonfile.write(json.dumps(tempjson))
#         jsonfile.close()
#     # vec = s2v("Pracking list of Printed Circuit Board")


# 经过PCA
if __name__ == '__main__':
    logger.info('loading GoogleNews-vectors-negative300.bin')
    s2v = Sentence2Vec("./embedding/GoogleNews-vectors-negative300.bin")

    folder_path = "/home/videt/Projects/document_analysis/data/layout/test/gt"

    pca = PCA(n_components=128)
    flag = 0
    n = 0
    for json_name in os.listdir(folder_path):
        logger.info('processing image{}'.format(n))
        n += 1
        jsonfile = open(os.path.join(folder_path,json_name), encoding="utf-8")
        jsonlist = jsonfile.read()
        tempjson = json.loads(jsonlist, encoding="utf-8")
        jsonfile.close()
        for idx,item in enumerate(tempjson):
            print(item["rec"])
            # logger.info(idx)
            if not item["rec"]:
                tempvec = [np.zeros([300]).tolist()]
            else:
                tempvec = s2v([item["rec"]]).tolist()
                # tempvec = [np.zeros([768]).tolist()]
            tempvec = np.array(tempvec)
            if flag == 0:
                tempveclist = tempvec
                flag = 1
            else:
                tempveclist = np.concatenate((tempveclist, tempvec))

    pca.fit(tempveclist)
    tempveclist = pca.transform(tempveclist)
    logger.info(tempveclist.shape)

    n = 0
    i = 0
    for json_name in os.listdir(folder_path):
        logger.info('writing embedding image{}'.format(n))
        n += 1
        jsonfile = open(os.path.join(folder_path,json_name), encoding="utf-8")
        jsonlist = jsonfile.read()
        tempjson = json.loads(jsonlist, encoding="utf-8")
        jsonfile.close()
        for idx,item in enumerate(tempjson):
            tempjson[idx]["embedding"] = tempveclist[i].tolist()
            i += 1
        print(len(tempjson[0]["embedding"]))
        jsonfile = open(os.path.join(folder_path,json_name), "w")
        jsonfile.write(json.dumps(tempjson))
        jsonfile.close()
    logger.info(tempveclist.shape)
    logger.info(i)



