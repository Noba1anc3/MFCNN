# -*- coding: utf-8 -*-
# Copyright(c) 2018-present, Videt Tech. All rights reserved.
# @Project : MSR
# @Time    : 19-2-18 下午4:26
# @Author  : kongshuchen
# @FileName: transforms.py
# @Software: PyCharm
import cv2
import numpy.random as random

from datasets.transforms.matrix_transform import *
from logzero import logger


def expand_image(img, new_h, new_w):
    """
     将img向h，w方向扩展为ks的最小倍数+ks
    :param img:
    :param new_h:
    :param new_w:
    :return:
    """
    h, w = img.shape[:2]
    assert new_h >= h and new_w >= w

    left = 0
    top = 0
    right = new_w - w
    bottom = new_h - h

    ret = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT)
    return ret

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts=None):
        for t in self.transforms:
            img, pts = t(img, pts)
        return img, pts


class RandomMirror(object):
    def __init__(self):
        pass

    def __call__(self, image, boxes=None):
        if np.random.randint(2):
            image = np.ascontiguousarray(image[:, ::-1])
            _, width, _ = image.shape
            boxes[:, :, 0] = width - boxes[:, :, 0]
        return image, boxes


class AugmentColor(object):
    def __init__(self):
        self.U = np.array([[-0.56543481, 0.71983482, 0.40240142],
                           [-0.5989477, -0.02304967, -0.80036049],
                           [-0.56694071, -0.6935729, 0.44423429]], dtype=np.float32)
        self.EV = np.array([1.65513492, 0.48450358, 0.1565086], dtype=np.float32)
        self.sigma = 0.1
        self.color_vec = None

    def __call__(self, img, boxes):
        color_vec = self.color_vec
        if self.color_vec is None:
            if not self.sigma > 0.0:
                color_vec = np.zeros(3, dtype=np.float32)
            else:
                color_vec = np.random.normal(0.0, self.sigma, 3)

        alpha = color_vec.astype(np.float32) * self.EV
        noise = np.dot(self.U, alpha.T) * 255
        return np.clip(img + noise[np.newaxis, np.newaxis, :], 0, 255), boxes


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return np.clip(image, 0, 255), boxes


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes):
        image = image.astype(np.float32)
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return np.clip(image, 0, 255), boxes


class Rotate(object):
    def __init__(self):
        pass

    def __call__(self, img, pts=None):
        h, w = img.shape[:2]
        dst_h, dst_w = h, w
        if random.random() > 0.5:
            # angle = random.randint(0, 179) - 90
            if random.random() > 0.9:
                if random.random() > 0.5:
                    angle = 90
                else:
                    angle = -90
            else:
                angle = random.randint(0, 360 - 1) - 180
                # print('rotate', angle)
            center = (int(dst_w / 2), int(dst_h / 2))
            M2 = genRotateAroundMatrix(center, angle, dst_w, dst_h)

            vertex = np.array([[0, 0],
                               [dst_w - 1, 0],
                               [dst_w - 1, dst_h - 1],
                               [0, dst_h - 1]], dtype=np.float32)

            new_vertex = warpQuadByMatrix(vertex, M2)
            diff = np.max(new_vertex, axis=0) - np.min(new_vertex, axis=0)
            dst_h, dst_w = int(diff[1]), int(diff[0])
        else:
            M2 = np.eye(3, dtype=np.float32)

        if pts is not None:
            new_pts = warpQuadByMatrix(pts, M2)

        if random.random() > 0.5:
            new_image = warpImageByMatrix(img, M2, dst_w, dst_h, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            new_image = warpImageByMatrix(img, M2, dst_w, dst_h, borderMode=cv2.BORDER_REPLICATE, borderValue=None)
        new_pts[..., 0] = np.clip(new_pts[..., 0], 0, dst_w - 1)
        new_pts[..., 1] = np.clip(new_pts[..., 1], 0, dst_h - 1)
        return new_image, new_pts


class SquarePadding(object):

    def __call__(self, image, pts=None):

        H, W, _ = image.shape

        if H == W:
            return image, pts

        padding_size = max(H, W)
        expand_image = np.zeros((padding_size, padding_size, 3), dtype=image.dtype)

        if H > W:
            y0, x0 = 0, (H - W) // 2
        else:
            y0, x0 = (W - H) // 2, 0
        if pts is not None:
            pts[:, 0] += x0
            pts[:, 1] += y0

        expand_image[y0:y0 + H, x0:x0 + W] = image
        image = expand_image

        return image, pts


class Padding(object):

    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, image, pts):
        if np.random.randint(2):
            return image, pts

        height, width, depth = image.shape
        ratio = np.random.uniform(1, 2)
        left = np.random.uniform(0, width * ratio - width)
        top = np.random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.fill
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image

        pts[:, 0] += left
        pts[:, 1] += top
        return image, pts


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.3, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = np.random.uniform(*scale) * area
            aspect_ratio = np.random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if np.random.random() < 0.5:
                w, h = h, w

            if h < img.shape[0] and w < img.shape[1]:
                j = np.random.randint(0, img.shape[1] - w)
                i = np.random.randint(0, img.shape[0] - h)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, image, pts=None):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        cropped = image[i:i + h, j:j + w, :]
        pts = pts.copy()
        mask = (pts[:, 1] >= i) * (pts[:, 0] >= j) * (pts[:, 1] < (i + h)) * (pts[:, 0] < (j + w))
        pts[~mask, 2] = -1
        scales = np.array([self.size[0] / w, self.size[1] / h])
        pts[:, :2] -= np.array([j, i])
        pts[:, :2] = (pts[:, :2] * scales)
        img = cv2.resize(cropped, self.size)
        return img, pts


class RandomResizedLimitCrop(object):
    def __init__(self, size, scale=(0.3, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = np.random.uniform(*scale) * area
            aspect_ratio = np.random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if np.random.random() < 0.5:
                w, h = h, w

            if h < img.shape[0] and w < img.shape[1]:
                j = np.random.randint(0, img.shape[1] - w)
                i = np.random.randint(0, img.shape[0] - h)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, image, pts):
        num_joints = np.sum(pts[:, -1] != -1)
        attempt = 0
        scale_vis = 0.75
        while attempt < 10:
            i, j, h, w = self.get_params(image, self.scale, self.ratio)
            mask = (pts[:, 1] >= i) * (pts[:, 0] >= j) * (pts[:, 1] < (i + h)) * (pts[:, 0] < (j + w))
            if np.sum(mask) >= (round(num_joints * scale_vis)):
                break
            attempt += 1
        if attempt == 10:
            w = min(image.shape[0], image.shape[1])
            h = w
            i = (image.shape[0] - w) // 2
            j = (image.shape[1] - w) // 2

        cropped = image[i:i + h, j:j + w, :]
        pts = pts.copy()
        mask = (pts[:, 1] >= i) * (pts[:, 0] >= j) * (pts[:, 1] < (i + h)) * (pts[:, 0] < (j + w))
        pts[~mask, 2] = -1
        scales = np.array([self.size[0] / w, self.size[1] / h])
        pts[:, :2] -= np.array([j, i])
        pts[:, :2] = (pts[:, :2] * scales).astype(np.int)
        img = cv2.resize(cropped, self.size)
        return img, pts


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, boxes=None):
        image = image.astype(np.float32)
        image /= 255.0
        # image -= self.mean
        # image /= self.std
        return image, boxes

class Align(object):
    def __init__(self):
        self.ratio = 16.

    def __call__(self,image,boxes=None):
        h, w = image.shape[:2]
        expanded_h = int(math.ceil(h / self.ratio) * self.ratio)
        expanded_w = int(math.ceil(w / self.ratio) * self.ratio)
        expanded_img = expand_image(image, expanded_h, expanded_w)
        return expanded_img, boxes


class Scale(object):
    def __init__(self):
        self.max_h = 600
        self.max_w = 600

    def __call__(self, image, boxes=None):
        h, w = image.shape[:2]
        if random.random() > 0.5:
            scale = 0.5
            # scale = min(random.uniform(0, 1), min(math.ceil(self.max_w / w), math.ceil(self.max_h / h)))  # 不大于最大长度
        else:
            # scale = min(1, min(math.ceil(self.max_w / w), math.ceil(self.max_h / h)))  # 不大于最大长度
            scale = 0.5
        dst_h, dst_w = int(scale * h), int(scale * w)
        image = cv2.resize(image, (dst_w, dst_h))
        expanded_h = int(math.ceil(dst_h / 16.0) * 16)
        expanded_w = int(math.ceil(dst_w / 16.0) * 16)
        expanded_img = expand_image(image, expanded_h, expanded_w)
        if boxes is not None:
            boxes = np.array(boxes * scale, dtype=int)

        return expanded_img, boxes

class Resize(object):
    def __init__(self, ratio):
        self.ratio = ratio
    
    def __call__(self, image, boxes=None):
        h, w, _ = image.shape
        dst_h, dst_w = int(self.ratio * h), int(self.ratio * w)
        image = cv2.resize(image, (dst_w, dst_h))
        expanded_h = int(math.ceil(dst_h / 16.0) * 16)
        expanded_w = int(math.ceil(dst_w / 16.0) * 16)
        expanded_img = expand_image(image, expanded_h, expanded_w)
        if boxes is not None:
            boxes = np.array(boxes * scale, dtype=int)
        return expanded_img, boxes

class Augmentation(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            Rotate(),
            Scale(),
            RandomMirror(),
            Normalize(mean, std)
        ])

    def __call__(self, image, boxes=None):
        return self.augmentation(image, boxes)


class BaseTransform(object):
    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            Align(),
            Normalize(mean, std)
        ])

    def __call__(self, image, boxes=None):
        return self.augmentation(image, boxes)
