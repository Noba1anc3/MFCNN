from bert_serving.client import BertClient
from sklearn.decomposition import PCA

import numpy as np

class DefpSentence2Vec(object):

    def __call__(self, sentence):
        bc = BertClient()
        vec = bc.encode(sentence)
        return vec


pca = PCA(n_components=12)
s2v = DefpSentence2Vec()

l = np.array(s2v(['拾元']))
t = ['贰拾元', '四川省成都市', '浙江省杭州市', '中国', '发票', '15107158D003', '00069445',  # 1-7
     '510798775802711', '15107179E001', '00029467', '510798775802711',  # 8 - 11
     '四川顺丰速运有限公司绵阳分公司通用定额发票', '都全程德邦物资有限公司通用定额发票', '发票查询网址:http://www.sc-n-txgv.cn', '发票查询网址:ht://www.sc-n-tax.gov.cn']   # 12-15
for i in t:
    tmp = np.array(s2v([i]))
    l = np.concatenate((l, tmp))
lc = l
ld = l[:,:128]
pca.fit(l)
l = pca.transform(l)
# print('a*b', np.dot(a, b.T))
# print('a*c', np.dot(a, c.T))
# print('c*d', np.dot(c, d.T))
# print('c*e', np.dot(c, e.T))
# print('c*f', np.dot(c, f.T))
# print('a*b', np.dot(l[0], l[1].T))  # pca 12
# print('a*g', np.dot(l[0], l[6].T))
# print('h*i', np.dot(l[6], l[9].T))
# print('g*j', np.dot(l[6], l[7].T))
# print('g*h', np.dot(l[6], l[8].T))
# print('h*k', np.dot(l[7], l[10].T))
# print('\n')
# print('a*b', np.dot(lc[0], lc[1].T))  # 原向量 768
# print('a*g', np.dot(lc[0], lc[6].T))
# print('h*i', np.dot(lc[6], lc[9].T))
# print('g*j', np.dot(lc[6], lc[7].T))
# print('g*h', np.dot(lc[6], lc[8].T))
# print('h*k', np.dot(lc[7], lc[10].T))
# print('\n')
# print('a*b', np.dot(ld[0], ld[1].T))   # 前128
# print('a*g', np.dot(ld[0], ld[6].T))
# print('h*i', np.dot(ld[6], ld[9].T))
# print('g*j', np.dot(ld[6], ld[7].T))
# print('g*h', np.dot(ld[6], ld[8].T))
# print('h*k', np.dot(ld[7], ld[10].T))
print('a*b', np.dot(lc[0], lc[12].T))  # 原向量 768
print('a*g', np.dot(lc[12], lc[13].T))
print('h*i', np.dot(lc[14], lc[15].T))
print('g*j', np.dot(lc[12], lc[14].T))
