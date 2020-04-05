import json
import numpy as np
import torch
import os
from configs import cfg
import errno


def mkdirs(newdir):
    try:
        if not os.path.exists(newdir):
            os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise

def str2bool(s):
    return s.lower() in ('true', '1')


def save_checkpoint(epoch, net_state_dict, optimizer_state_dict, best_score, checkpoint_path, model_path):
    torch.save({
        'epoch': epoch,
        'model': net_state_dict,
        'optimizer': optimizer_state_dict,
        'best_score': best_score
    }, checkpoint_path)
    torch.save(net_state_dict, model_path)


def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False


def load_json(json_path, phase='train'):
    fi = open(json_path, 'r', encoding='utf-8')
    json_content = fi.read()
    fi.close()
    json_content = json.loads(json_content, encoding='utf-8')
    boxes = []
    for item in json_content:
        if 'bbox' in item.keys():
            boxes.append(item['bbox'])
        elif 'locations' in item.keys():
            boxes.append(item['locations'])
    if phase =='val':
        value = []
        for item in json_content:
            value.append(item['value'])
        return np.array(boxes, dtype=np.int32), value
    return np.array(boxes, dtype=np.int32)


def load_txt(txt_path, phase='train'):
    fi = open(txt_path, 'r', encoding='utf-8')
    txt_content = fi.read().split("\n")
    fi.close()
    boxes = []
    for item in txt_content:
        if phase == 'train':
            gt_idx = item.split(" ")
        else:
            gt_idx = item.split(", ")
        if len(gt_idx) > 1:
            boxes.append([
                [int(gt_idx[0]), int(gt_idx[1])],
                [int(gt_idx[0]), int(gt_idx[3])],
                [int(gt_idx[2]), int(gt_idx[3])],
                [int(gt_idx[2]), int(gt_idx[1])]
            ])
    if phase =='val':
        value = []
        for item in txt_content:
            value.append([])
        return np.array(boxes, dtype=np.int32), value
    return np.array(boxes, dtype=np.int32)


def getImgPath(root_path, relative_path, path_list=[]):
    if os.path.isfile(os.path.join(root_path, relative_path)):
        if relative_path[-4:] == '.jpg' or relative_path[-4:] == '.png':
            path_list.append(relative_path)
    else:
        for name in os.listdir(os.path.join(root_path, relative_path)):
            sub_path = os.path.join(relative_path, name)
            getImgPath(root_path, sub_path, path_list)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_device(*tensors):
    return (t.to(cfg.MODEL.DEVICE) for t in tensors)


def mkdirs(newdir):
    try:
        if not os.path.exists(newdir):
            os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise


def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x ** 2, axis=axis))
    return np.sqrt(np.sum(x ** 2))
