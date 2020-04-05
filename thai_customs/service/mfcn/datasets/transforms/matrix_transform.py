"""
@Description:
@version: V1.0
@Company: VIDET
@Author: WUSHUFAN
@Date: 2019-02-27 19:19:13
"""
import math

import cv2
import numpy as np


def genShiftMatrix(src_point, dst_point):
    # shift_x, shift_y = src_point - dst_point
    shift_x, shift_y = dst_point - src_point
    M = np.array([[1, 0, shift_x],
                  [0, 1, shift_y],
                  [0, 0, 1]], dtype=np.float32)
    return M


def genRotateAroundMatrix(center, angle, w, h, return_new_size=False):
    origin = np.array([0, 0], dtype=np.float32)
    M1 = genShiftMatrix(center, origin)
    M2 = genRotateMatrix(angle)
    vertex = np.array([[0, 0],
                       [w - 1, 0],
                       [w - 1, h - 1],
                       [0, h - 1]], dtype=np.float32)
    new_vertex = warpQuadByMatrix(vertex, np.matmul(M2, M1))
    left, top = np.min(new_vertex[:, 0]), np.min(new_vertex[:, 1])
    right, bottom = np.max(new_vertex[:, 0]), np.max(new_vertex[:, 1])

    new_w, new_h = int(np.round(right - left + 1, 0)), int(np.round(bottom - top + 1, 0))

    M3 = genShiftMatrix(np.array([left, top], dtype=np.float32), origin)
    M = np.matmul(M3, np.matmul(M2, M1))

    if return_new_size:
        return M, new_w, new_h
    else:
        return M


def genRotateMatrix(angle):
    '''
    angle是角度
    :param angle:
    :return:
    '''
    angle1 = angle * np.pi / 180
    a = math.cos(angle1)
    b = math.sin(angle1)

    M = np.array([[a, -b, 0],
                  [b, a, 0],
                  [0, 0, 1]], dtype=np.float32)
    return M


def warpImageByMatrix(img, M, w, h, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE, borderValue=None):
    if type(borderValue) == type(None):
        img1 = cv2.warpPerspective(img, M, (w, h), flags=flags, borderMode=borderMode)
    else:
        img1 = cv2.warpPerspective(img, M, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue)

    # img1 = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return img1


def warpQuadByMatrix(quad, M):
    quad_shape = quad.shape

    if len(quad_shape) < 2:
        return quad
    if len(quad_shape) == 2:
        quad = quad[np.newaxis, :, :]

    quad_proj = np.concatenate([quad.astype(np.float32), np.ones((quad.shape[0], quad.shape[1], 1), dtype=np.float32)],
                               axis=-1)
    quad_proj = quad_proj.transpose(0, 2, 1)
    M = np.broadcast_to(M, (quad.shape[0], 3, 3))
    rotated_quad_proj = np.matmul(M, quad_proj)
    rotated_quad = rotated_quad_proj[:, :2, :].transpose(0, 2, 1)  # .astype(np.int32)

    if len(quad_shape) == 2:
        rotated_quad = rotated_quad[0, :, :]
    return rotated_quad
