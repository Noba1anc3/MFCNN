import time
from logzero import logger

def read_info(item):
    reco = item["value"]
    box = item["locations"][0]
    return box, reco

def json_format_change(dr_to_json, keylist):

    start_time = time.time()
    json_str = dr_to_json["single"][0]["res"]["items"]
    clean_json = []

    for item in json_str:
        for key_num in range(0, len(keylist)):
            if keylist[key_num] in item:
                cls_type = key_num + 1
                box, reco = read_info(item[keylist[key_num]])
                clean_json.append({"det": box, "rec": reco, "cls": cls_type})

    logger.info("**** Json Format Changed: {}ms".format('%.4f'%((time.time() - start_time)*1000)))

    return clean_json


