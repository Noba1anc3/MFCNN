
import sys
import json
import time
import torch
import argparse
sys.path.append('../')

from utils.dynamic_load import build_loss, build_model, build_dataset
from datasets.inference_dataset import InferenceDataset
from agents.inference_predict import out_test
from logzero import logger
import v_thai_customs.service

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "opts",
        help = "Modify config options using the command-line",
        default = None,
        nargs = argparse.REMAINDER,
    )
    parser.add_argument("--eval", default = True)
    parser.add_argument("--embedding_dim", type = int, default = 128)
    parser.add_argument("--with_embedding", default = True)
    parser.add_argument("--embedding_first_layer", default = True)
    parser.add_argument("--dilation", type = int, default = 3)
    args = parser.parse_args()

    return args

def read_config_json(json_path):
    json_file = open(json_path, encoding="utf-8")
    json_list = json_file.read()
    temp_json = json.loads(json_list, encoding="utf-8")
    json_file.close()
    return temp_json

def load_model(cfgs):
    start_time = time.time()
    model_inv = build_model(cfgs, 1)
    checkpoint_inv = torch.load(v_thai_customs.service.__file__[:-12] + '/mfcn/models/' + cfgs.model_name_inv)
    model_inv.load_state_dict(checkpoint_inv)
    model_inv.to('cuda:0')
    model_inv.eval()

    model_pl = build_model(cfgs, 2)
    checkpoint_pl = torch.load(v_thai_customs.service.__file__[:-12] + '/mfcn/models/' + cfgs.model_name_pl)
    model_pl.load_state_dict(checkpoint_pl)
    model_pl.to('cuda:0')
    model_pl.eval()

    logger.info("------------- MFCN Model Loaded  Time Consume:{}ms --------------".format('%.0f'%((time.time() - start_time)*1000)))

    return model_inv, model_pl

def mfcn_inference(cfgs, img_type, img, gt, dt2json, keylist):

    InfData = InferenceDataset(img, gt, len(keylist))
    image, segment_gt, embedding_layer, boxes, label = InfData.get_inference_data()

    image = image.float().to('cuda:0')
    embedding_layer = embedding_layer.float().to('cuda:0')

    if img_type == 'Invoice':
        model = cfgs.model_inv
    else:
        model = cfgs.model_pl

    pred, _, _, _, _, _, _, _ = model(image, embedding_layer)

    return out_test(pred, boxes, segment_gt, dt2json, keylist)

def has_Commodity(array):
    if array.split('.')[0] == 'PLCommodity' or array.split('.')[0] == 'INVCommodity':
        return True
    else:
        return False

def json_pros(img_type, key_list, value_list, loc_list):
    if img_type == 'Invoice':
        img_type = 'invoice'
    elif img_type == 'PackingList':
        img_type = 'packing_list'

    res = {}
    in_dic = {}
    commodity = []
    for idx, key in enumerate(key_list):
        if has_Commodity(key):
            if key in in_dic.keys():
                commodity.append(in_dic)
                in_dic = {}
                in_dic[key] = {}
                in_dic[key]['value'] = value_list[idx]
                in_dic[key]['location'] = [loc_list[idx]]
            else:
                in_dic[key] = {}
                in_dic[key]['value'] = value_list[idx]
                in_dic[key]['location'] = [loc_list[idx]]
        elif key not in res.keys():
            res[key] = {}
            res[key]["value"] = value_list[idx]
            res[key]["location"] = [loc_list[idx]]
        else:
            res[key]["value"].join(value_list[idx])
            res[key]["location"].append(loc_list[idx])

    if len(in_dic) > 0:
        commodity.append(in_dic)
    if len(commodity) > 0:
        res['Commodity'] = commodity
    invoice_packinglist = {}
    invoice_packinglist[img_type] = res
    json_str = json.dumps(invoice_packinglist)

    return invoice_packinglist
