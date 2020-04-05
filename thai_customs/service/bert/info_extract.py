
from v_thai_customs.service.bert.data_process.json_format_change import json_format_change
from v_thai_customs.service.bert.data_process.bert_sen2vec import GenerateEmbeddingLayer
from v_thai_customs.service.bert.data_process.change_data_size import change_data_size

def info_extra(configs, img_type, image, dr_to_json, keylist):
    clean_json = json_format_change(dr_to_json, keylist)
    GT = GenerateEmbeddingLayer(clean_json)
    GT.vec768_generation(configs)
    json_vec128 = GT.vec768_to_128(configs, img_type)
    resized_image = change_data_size(image, json_vec128)

    return resized_image, json_vec128
