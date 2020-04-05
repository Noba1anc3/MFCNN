import cv2
import numpy as np
import torch
import torch.nn.functional as F
from datasets.inference_dataset import colorimg
from v_dar_tools.box_tool.sort_box import sort_box_by_row

def triarea(a,b,c):
    return 0.5*(a[0]*b[1]+b[0]*c[:,1]+c[:,0]*a[1]-a[0]*c[:,1]-b[0]*a[1]-c[:,0]*b[1])

def within(points, b):
    vis = np.ones((len(points),1))
    for i in range(b.shape[0]):
        begin,end = b[i],b[(i+1)%b.shape[0]]
        area = triarea(begin,end,points)
        vis[np.where(area<0)] = 0
        if i % 2 == 0:
            vis[np.where(np.abs(area)<1e-5)] = 0
    return vis

def out_test(pred_image, boxes, segment_gt, json_file, keylist):

    rec_result = []
    loc_result = []

    tempjson = json_file["single"][0]["res"]["items"]
    for i in range(len(tempjson)):
        for item in tempjson[i]:
            rec_result.append(tempjson[i][item]["value"])
            loc_result.append(tempjson[i][item]["locations"][0])

    _, C, H, W = pred_image.size()

    image = pred_image.permute(0, 2, 3, 1).view(-1, len(keylist) + 1).cpu()
    pred_image = torch.nn.functional.softmax(image)
    score, pred_map = torch.max(pred_image, 1)
    pred_map = pred_map.view(H, W)
    image = pred_map.detach().numpy()
    image = np.array(image, dtype=np.uint8)

    boxes = boxes.numpy()[0]
    index_in_boxes, _ = sort_box_by_row(boxes)

    gt = segment_gt[0]
    gt = gt.detach().cpu().numpy()
    pointx, pointy = np.where(gt > 0)
    pointx = pointx.reshape((pointx.shape[0], 1))
    pointy = pointy.reshape((pointy.shape[0], 1))
    points = np.concatenate((pointy, pointx), 1)

    value_list = []
    loc_list = []
    key_list = []

    for row in index_in_boxes:
        for col in range(len(row)):
            index = row[col]
            box = boxes[index]
            box = np.array(box)
            postive_idx = within(points, box)
            postive_point = points[np.where(postive_idx > 0)[0], :]
            postive_y = postive_point[:, 0]
            postive_x = postive_point[:, 1]
            predict_label = image[postive_x, postive_y]
            pred_label = [0 for i in range(len(keylist) + 1)]
            for i in range(0, len(keylist) + 1):
                pred_label[i] = np.sum(predict_label == i)
            pred_cls = np.argmax(pred_label)

            if not pred_cls == 0 and not pred_cls == 1:
                key_list.append(keylist[pred_cls - 1])
                value_list.append(rec_result[index])
                loc_list.append(loc_result[index])

    return key_list, value_list, loc_list
