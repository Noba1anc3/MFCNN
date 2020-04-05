from datasets.transforms import transforms


class TrainAugmentation:
    def __init__(self, cfg):
        aug_list = []
        aug_list.append(transforms.Normalize(mean=cfg.DATASET.NORM[0], std=cfg.DATASET.NORM[1]))
        self.augment = transforms.Compose(aug_list)

    def __call__(self, img, boxes=None):
        """
        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
        """
        return self.augment(img, boxes)


class TestTransform:
    def __init__(self, cfg):
        trans_list = [getattr(transforms, aug_item)() for aug_item in cfg.DATASET.TEST_TRANS]
        trans_list.append(transforms.Normalize(cfg.DATASET.NORM))
        self.transform = transforms.Compose([trans_list])

    def __call__(self, image):
        return self.transform(image)
