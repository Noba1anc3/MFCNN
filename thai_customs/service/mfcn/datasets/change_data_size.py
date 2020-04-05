import numpy as np
import cv2


def change_data_size(image, gt):

    tempjson = gt
    max_long = 256
    if max_long < min(image.shape[0], image.shape[1]):
        ratio = max_long / min(image.shape[0], image.shape[1])
        resize_image = cv2.resize(image, None, None, ratio, ratio,interpolation = cv2.INTER_CUBIC)
        for idx,item in enumerate(tempjson):
            box = np.array(item["det"])
            box = np.array(box*ratio,dtype=np.int32).tolist()
            tempjson[idx]["det"] = box
    else:
        resize_image = image
    return resize_image, tempjson
