
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from logzero import logger


class WeightCELoss(nn.Module):
    def __init__(self):
        super(WeightCELoss, self).__init__()

    def forward(self, pred, gt, w):

        WCELoss = F.cross_entropy(pred, gt, weight=w)
        return WCELoss


class OhemCELoss(nn.Module):
    def __init__(self):
        super(OhemCELoss, self).__init__()

    def forward(self, cfg, pred, gt, negative_ratio=3):
        predict = pred.permute(0, 2, 3, 1).view(-1, cfg.MODEL.NUM_CLASSES)
        target = gt.view(-1)
        mask = target.cpu().numpy()
        mask[mask == 4] = 0
        mask[mask == 2] = 1
        mask[mask == 3] = 1
        target1 = torch.from_numpy(mask).cuda()
        pos = target1.byte()
        neg = (1 - target1).byte()

        n_pos = pos.float().sum()
        n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))

        loss_pos = F.cross_entropy(predict[pos], target[pos], size_average=False)
        loss_neg = F.cross_entropy(predict[neg], target[neg], reduce=False)
        loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()


class Rec_Loss(nn.Module):
    def __init__(self):
        super(Rec_Loss, self).__init__()

    def forward(self, a_rec, a):
        MSELoss = nn.MSELoss(a_rec, a)
        return MSELoss


# class Cons_Loss_pre(nn.Module):
#     def __init__(self):
#
#         super(Cons_Loss, self).__init__()
#
#     def forward(self, pred, gt1, boxes):  # , image, gt
#
#         suml = 0
#
#         for l in range(0, 3):
#
#             for i in range(boxes + 1):
#                 mask = gt1[0, l, ::] == i
#                 mask = mask.float()
#                 pb = gt1[0, l, ::].clone()
#                 # pb = torch.ones(gt1[0, l, ::].shape).cuda()
#
#                 pb[gt1[0, l, ::] == i] = torch.mean(pred[0, l, ::][gt1[0, l, ::] == i])
#
#                 temp_pred = pred[0, l, ::].mul(mask)
#                 pb_temp = pb.mul(mask)
#                 temp_sub = temp_pred - pb_temp
#
#                 temp_sum = torch.sum(torch.pow(temp_sub, 2))
#                 b = torch.add(torch.sum(mask), torch.Tensor([1]).cuda())
#                 # print(temp_sum)
#                 # print(b)
#                 suml += torch.div(temp_sum, b)
#         return suml


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


class Cons_Loss(nn.Module):
    def __init__(self):

        super(Cons_Loss, self).__init__()

    def forward(self, pred, gt, boxes):
        predcopy = pred.clone()
        predcopy = predcopy.detach().cpu().numpy()

        gt = gt.detach().cpu().numpy()

        pointx, pointy = np.where(gt[0, :, :] > 0)
        pointx = pointx.reshape((pointx.shape[0], 1))
        pointy = pointy.reshape((pointy.shape[0], 1))
        points = np.concatenate((pointy, pointx), 1)
        sumloss = 0
        for l in range(0, predcopy.shape[1]):
            for idx, box in enumerate(boxes):
                for i in range(0, box.shape[0]):
                    box = np.array(box)
                    # cv2.fillPoly(embedding_feature,box,embedding[idx],1)
                    postive_idx = within(points, box[i])
                    postive_point = points[np.where(postive_idx > 0)[0], :]
                    postive_y = postive_point[:, 0]
                    postive_x = postive_point[:, 1]
                    sub_prednp = predcopy[0, l, ::][postive_x, postive_y]

                    temp_mean = torch.Tensor([np.mean(sub_prednp)]).cuda()
                    if len(sub_prednp) == 0:
                        break
                    temp_div = torch.Tensor([len(sub_prednp)]).cuda()
                    temp_sub = pred[0, l, ::][postive_x, postive_y] - temp_mean
                    temp_sum = torch.sum(torch.pow(temp_sub, 2))

                    sumloss += torch.div(temp_sum, temp_div)

        return sumloss


if __name__ == '__main__':

    dtype = torch.float
    device = torch.device("cuda:0")
    N, D_in, H, D_out = 1, 3, 4, 4

    a = [[[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                         [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                         [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]]]

    pred = torch.Tensor(a)
    pred.requires_grad = True

    b = [[[0, 0,0,0], [0, 1, 1, 1], [0, 1,1,1], [2, 0, 0, 0]]]
    gt1 = torch.Tensor(b)
    gt1.requires_grad = True
    boxes = [[1,0],[3,0],[3,3],[1,3]]

    l = Cons_Loss()
    loss = l(pred, gt1, boxes)

    logger.info(loss)
    loss.backward()


def create_loss():
    return 0
