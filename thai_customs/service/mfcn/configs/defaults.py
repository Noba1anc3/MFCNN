from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = ''
_C.MODEL.LOSS = ''
_C.MODEL.RATIO = 16
_C.MODEL.DEVICE = "cuda:0"
_C.MODEL.NUM_CLASSES = 2

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = ''
_C.DATASET.ROOT_DIR = ''
_C.DATASET.TYPE = ''
_C.DATASET.image_format = ''
_C.DATASET.TRAIN_AUG = []
_C.DATASET.TEST_TRANS = []
_C.DATASET.NORM = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
_C.DATASET.MAX_H = 2880
_C.DATASET.MAX_W = 2880
_C.DATASET.IMAGE_SIZE = 512

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# train configs
_C.SOLVER.MAX_ITER = 120000
_C.SOLVER.LR_STEPS = [80000, 100000]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.LR = 1e-3
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500

_C.OUTPUT_DIR = './output'
