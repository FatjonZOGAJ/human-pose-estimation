import os
import logging

import torch
import torch.nn as nn
from lib.utils.integral import softmax_integral_tensor

import numpy as np

logger = logging.getLogger(__name__)

class PoseAlexNetReg(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(PoseAlexNetReg, self).__init__()
        self.cfg = cfg
        #self.c = torch.nn.Parameter(torch.rand(1)-torch.rand(1))
        #self.register_parameter(name='corr_factor', param=self.c)

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Results in 32 units, representing a 10 meter range in front of the camera
        self.conv_1D = nn.Sequential(nn.Conv1d(1, 1, kernel_size=225), nn.ReLU(inplace=True))
        self.conv_2D = nn.Sequential(nn.Conv2d(256, self.cfg.MODEL.NUM_JOINTS * self.cfg.MODEL.NUM_DEPTH, kernel_size=1), nn.ReLU(inplace=True))
        self.volume_pipe_1D = self.heat_1D_pipe
        self.volume_pipe_3D = self.heat_3D_pipe


    def forward(self, x):
        x = self.features(x)
        x_1D = self.volume_pipe_1D(x)
        x_3D = self.volume_pipe_3D(x)
        #back_proj = self.back_project(x_1D, x_3D)
        return torch.reshape(x_3D, [x.shape[0],-1])

    # Returns Z*
    def heat_1D_pipe(self, x: torch.Tensor):
        x = x.mean([2,3])
        x = x.unsqueeze(1)
        x = self.conv_1D(x)
        return softmax_integral_tensor(x, 1, 1, 1, 1, x.shape[2])

    # Returns x, y, delta Z
    def heat_3D_pipe(self, x: torch.Tensor):
        x = self.conv_2D(x)
        x = torch.reshape(x, [x.shape[0], self.cfg.MODEL.NUM_JOINTS, self.cfg.MODEL.NUM_DEPTH, *x.shape[2:]])
        hm_d, hm_w, hm_h = x.shape[2:]
        return softmax_integral_tensor(x, self.cfg.MODEL.NUM_JOINTS, 3, hm_w, hm_h, hm_d)

    def back_project(self, x_1D, x_3D):
        pass

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained imagenet model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseAlexNetReg(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
