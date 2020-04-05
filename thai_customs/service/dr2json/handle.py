'''
@Description: 
@version: 
@Company: VIDET
@Author: ZHOUZHAO
@Date: 2019-01-21 17:34:20
@LastEditors: ZHANGXUANRUI
@LastEditTime: 2019-06-06 10:30:29
'''

import os
import cv2
import copy
import time
import itertools
import numpy as np
from logzero import logger
from v_dar_tools.business_tool.business_logic import BusinessLogic
import v_thai_customs.service

class Handle(BusinessLogic):
    def __init__(self, business_config):
        BusinessLogic.__init__(self, business_config)
        self.folder_prefix = 'dr2json/'
        self.ret_json = {}

    def prepare_data(self, temp_path, img_path, image_name_list, image_total_num):
        if not os.path.exists(os.path.join(temp_path, 'img')):
            os.makedirs(os.path.join(temp_path, 'img'))
        if not os.path.exists(os.path.join(temp_path, 'rec')):
            os.makedirs(os.path.join(temp_path, 'rec'))
        if not os.path.exists(os.path.join(temp_path, 'det')):
            os.makedirs(os.path.join(temp_path, 'det'))

        for idx, image_name in enumerate(image_name_list):
            if os.path.exists(os.path.join(temp_path, 'rec', image_name.split('.')[0] + '.npy')):
                continue

            image = cv2.imread(os.path.join(img_path, image_name))
            reco_result, flip_bboxes = self._prepare_data(image)
            reco_result = np.array(reco_result)
            flip_bboxes = np.array(flip_bboxes)

            cv2.imwrite(os.path.join(temp_path, 'img', image_name), image)
            np.save(os.path.join(temp_path, 'rec', image_name.split('.')[0] + '.npy'), reco_result)
            np.save(os.path.join(temp_path, 'det', image_name.split('.')[0] + '.npy'), flip_bboxes)
            logger.info(image_name + "  " + str(idx + 1) + "/" + str(image_total_num))

    def layout_extract(self):
        layout_config = self.read_config_json(v_thai_customs.service.__file__[:-11] + self.folder_prefix + 'layout_config_thai.json')
        thresholds_keys = self.read_config_json(v_thai_customs.service.__file__[:-11] + self.folder_prefix + 'thresholds_keys_thai.json')
        self._extract_info_by_key_position(layout_config, thresholds_keys)

    def ignore_info(self, key='ignore'):
        res_recos, res_boxes, index_in_boxes = self._get_sub_info_recos_by_key(key)
        index_in_boxes = np.array(list(itertools.chain.from_iterable(index_in_boxes[:])))
        if len(index_in_boxes) > 0:
            for index in index_in_boxes:
                sorted_boxes = res_boxes[index]
                sorted_recos = res_recos[index].strip()
                self._set_key_value_locations(key + str(index), sorted_recos, [sorted_boxes])

    def check_json(self):
        try:
            self.ret_json["single"] = []
            res = {}
            self.ret_json["single"].append(res)

            res["res"] = {}
            res["res"]["items"] = []

            packing_list_json = {}

            i = 0
            for key in self.final_result:
                if isinstance(self.final_result[key], list):
                    continue
                packing_list_json[key] = copy.copy(self.final_result[key])
                res["res"]["items"].append({})
                res["res"]["items"][i]["ignore"] = packing_list_json[key]
                i += 1
        except Exception:
            return {}

    def det_rec_to_json(self, image, flip_bboxes, reco_result, image_type = None):
        start_time = time.time()
        self.__prepare__(image, flip_bboxes, reco_result, image_type = image_type)
        self.layout_extract()
        self.ignore_info()
        self.check_json()

        logger.info("********************* JSON Generated: " + "{}ms".format('%.3f'%((time.time() - start_time)*1000)))

        return self.ret_json

    def read_info(self, image_name, folder_path = None):
        image = cv2.imread(os.path.join(folder_path, 'img', image_name))
        flip_bboxes = np.load(os.path.join(folder_path, 'det', image_name.split('.')[0] + '.npy'))
        reco_result = np.load(os.path.join(folder_path, 'rec', image_name.split('.')[0] + '.npy'))
        return image, flip_bboxes, reco_result
