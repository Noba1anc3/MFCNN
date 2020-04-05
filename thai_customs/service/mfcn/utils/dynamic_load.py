"""
@Description:
@version: V1.0
@Company: VIDET
@Author: WUSHUFAN
@Date: 2019-03-02 16:36:28
"""
import importlib


def build_loss(cfg):
    module = importlib.import_module('graphs.losses.{}'.format(cfg.MODEL.LOSS))
    return module.create_loss()

def build_dataset(cfg,args, transform, phase='train'):
    module = importlib.import_module('datasets.{}_dataset'.format(cfg.DATASET.NAME))
    return module.create_dataset(cfg, args, transform, phase)

def build_model(cfgs, inv_pl):
    args = cfgs.args
    module = importlib.import_module('graphs.models.{}'.format('unet'))
    return module.create_model(inv_pl, cfgs.inv_keylist, cfgs.pl_keylist, args)

def get_inferencer(method_name):
    module = importlib.import_module('agents.{}_inferencer'.format(method_name))
    return module

def get_trainer(method_name):
    module = importlib.import_module('agents.{}_trainer'.format(method_name))
    return module
