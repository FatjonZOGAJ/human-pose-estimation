import os
import logging

import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import model_urls

logger = logging.getLogger(__name__)


# TODO: take a look at keypoint specific architecture https://pytorch.org/vision/stable/models.html#keypoint-r-cnn
# for now use Classification specific regular ResNext as it's closest to AlexNet that was in use
class PoseResNextReg(nn.Module):

    def __init__(self, cfg, n_frozen_layers=9, **kwargs):
        super(PoseResNextReg, self).__init__()
        self.n_frozen_layers = n_frozen_layers
        self.model = models.resnext101_32x8d(pretrained=False,
                                             # load manually down below (optionally own pretrained data)
                                             num_classes=cfg.MODEL.NUM_JOINTS * 3,
                                             # num_classes like in adapted pose3d_alexnet_reg.py
                                             )
        logger.info(f'=> freezing ResNext101 first {self.n_frozen_layers} layers')
        for i, child in enumerate(self.model.children(), 1):
            for param in child.parameters():
                param.requires_grad = False
            if i == self.n_frozen_layers:
                break

    def forward(self, x):
        return self.model.forward(x)

    def init_weights(self, pretrained='', download_pretrained=True, arch='resnext101_32x8d'):
        if download_pretrained:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  model_dir='lib/models/pretrained_data',  # /checkpoints',
                                                  file_name='resnext_original.pth',
                                                  progress=True)
        elif os.path.isfile(pretrained):
            state_dict = torch.load(pretrained)
        else:
            logger.error('=> resnext pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('resnext pretrained model does not exist')

        logger.info('=> loading pretrained resnext model {}'.format(pretrained))
        self.load_state_dict(state_dict, strict=False)


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseResNextReg(cfg, **kwargs)

    # Some models use modules which have different training and evaluation behavior,
    # such as batch normalization. To switch between these modes, use model.train()
    # or model.eval() as appropriate. See train() or eval() for details.
    # model.train()
    if is_train:
        if cfg.MODEL.INIT_WEIGHTS:
            model.init_weights(cfg.MODEL.PRETRAINED, arch='resnext101_32x8d')
        model.train()
    else:
        model.eval()

    return model
