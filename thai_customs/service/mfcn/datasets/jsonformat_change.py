
from datasets.key_list import get_keylist


def read_info(it2):
    reco = it2["value"]
    box = it2["locations"][0]

    return box, reco


def json_format_change(keylist, json):
    yld_json = json["single"][0]

    save_json = []
    Commodity = yld_json["res"]
    Commodityitem = Commodity["items"]
    for it1 in Commodityitem:
        for key_num in range(0, len(keylist)):
            if keylist[key_num] in it1:

                cls_type = key_num + 1
                box, reco = read_info(it1[keylist[key_num]])
                save_json.append({"det": box, "rec": reco, "cls": cls_type})

    return save_json

