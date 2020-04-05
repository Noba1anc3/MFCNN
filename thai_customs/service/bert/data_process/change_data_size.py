import cv2
import time
import numpy as np
from logzero import logger

def change_data_size(image, json_vec128):
    start_time = time.time()
    max_long = 256
    if max_long < min(image.shape[0], image.shape[1]):
        ratio = max_long / min(image.shape[0], image.shape[1])
        resized_image = cv2.resize(image, None, None, ratio, ratio, interpolation = cv2.INTER_CUBIC)

        for idx,item in enumerate(json_vec128):
            box = np.array(item["det"])
            box = np.array(box * ratio, dtype = np.int32).tolist()
            json_vec128[idx]["det"] = box
    else:
        resized_image = image

    logger.info("** Resized Image Generated: {}ms".format('%.4f'%((time.time() - start_time)*1000)))

    return resized_image