# Research on Layout Analysis of Customs Declaration Bill

## Based on Rules
Aset of general rules for Information Extraction out of forms is designed and implemented independently.  
The accuracy rate of key-value pairs on more than 100 customs declaration invoices reaches 98%. 

## Based on Neural Network
MFCNN based on BERT is reproduced. A series of experiments are carried out in terms of training set requirements,  
training cost, model generalization, finetune for downstream tasks and parameter tuning.

### Learning to Extract Semantic Structure from Documents Using Multimodal Fully Convolutional Neural Networks
[Paper](https://arxiv.org/pdf/1706.02337)

### Documentation
- [Research Documentation on Layout Analysis](https://github.com/Noba1anc3/MFCN/wiki/Layout-Analysis)

### Model
- [Link](https://bhpan.buaa.edu.cn:443/link/9287EE12F3D262A1C62085F62A5DF5E1)
- 下载好的模型应放置于 mfcn 文件夹下的 models 文件夹内

### Auxiliary Files for PCA Dimension Reduction
- [Link](https://bhpan.buaa.edu.cn:443/link/4D32519306C601329547D672D714EA1A)
- 下载好的数据降维辅助文件应放置于 bert 文件夹下的 auxiliary_768 文件夹内

## Based on Machine Learning
The experiment of LightGBM Feature Engineering makes the F1 score of one shot learning exceed 0.9.
