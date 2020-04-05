import numpy as np
import cv2

colortable = np.array([[255, 255, 255], [255, 0, 0], [0, 255, 0]])
# gt = np.array([[0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0]])
gt = np.ones([5, 5])
# gt = gt.reshape(-1)
gt = gt.astype(int)
print(gt)
# colorgt = np.zeros([3, a, b])
colorgt = np.array(colortable[gt])
print(colorgt.shape)
print(colorgt)
gt = cv2.resize(gt,None,None,20,20)
cv2.imshow('img', gt)  # cv2.resize(colorgt,None,None,20,20)
cv2.waitKey(0)

