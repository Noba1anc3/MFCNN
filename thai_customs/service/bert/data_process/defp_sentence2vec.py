
import gensim
from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords as Stopwords
import numpy as np
import time
from sklearn.decomposition import PCA
from logzero import logger
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os
import json
from bert_serving.client import BertClient


class DefpSentence2Vec(object):

    def __call__(self, sentence):
        bc = BertClient()
        vec = bc.encode(sentence)
        return vec


if __name__ == '__main__':
    s2v = DefpSentence2Vec()

    folder_path = "/home/videt/Projects/document_analysis/data/defp/temp/gt"

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
                tempvec = [np.zeros([768]).tolist()]
            else:
                tempvec = s2v([item["rec"]]).tolist()
                # tempvec = [np.zeros([768]).tolist()]
            tempvec = np.array(tempvec)
            if flag == 0:
                tempveclist = tempvec
                flag = 1
            else:
                tempveclist = np.concatenate((tempveclist, tempvec))
                logger.info(tempvec.shape)
                logger.info(tempveclist.shape)

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


# 不经过 PCA
# if __name__ == '__main__':
#     s2v = DefpSentence2Vec()
#     folder_path = "/home/videt/Projects/document_analysis/data/defp/train/gt"
#     n = 0
#     for json_name in os.listdir(folder_path):
#         logger.info('image{}'.format(n))
#         n += 1
#         logger.info(json_name)
#         jsonfile = open(os.path.join(folder_path,json_name), encoding="utf-8")
#         jsonlist = jsonfile.read()
#         tempjson = json.loads(jsonlist, encoding="utf-8")
#         jsonfile.close()
#         for idx,item in enumerate(tempjson):
#             logger.info(idx)
#             logger.info(item["rec"])
#             if not item["rec"]:
#                 tempvec = [np.zeros([768]).tolist()]
#             else:
#                 tempvec = s2v([item["rec"]]).tolist()
#                 # tempvec = [np.zeros([768]).tolist()]
#             tempjson[idx]["embedding"] = tempvec
#         jsonfile = open(os.path.join(folder_path,json_name), "w")
#         jsonfile.write(json.dumps(tempjson))
#         jsonfile.close()


