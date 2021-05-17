import os
import logging

import torch
import torch.nn as nn

import torchvision.models

logger = logging.getLogger(__name__)


class CustomResNet50(torchvision.models.ResNet):
    def __init__(self):
        super(CustomResNet50, self).__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class PoseAlexNetReg(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(PoseAlexNetReg, self).__init__()
        self.volume_config = cfg.MODEL.VOLUME
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.depth_res = cfg.MODEL.DEPTH_RES
        self._make_backbone()
        self._make_head(self.num_joints, self.depth_res)
        self._softmax = torch.nn.Softmax(dim=2)
        self._range = torch.nn.Parameter(torch.arange(1, 64 + 1), requires_grad=False)

    def _make_backbone(self):
        self._backbone = CustomResNet50()
        #self.backbone.load_state_dict(torchvision.models.resnet50(pretrained=True).state_dict())
        del self._backbone.avgpool
        del self._backbone.fc

    def _make_head(self, num_joints, depth_res):
        layers = []
        kernel, padding, output_padding = self.volume_config.LAYER_CONFIG
        filters = self.volume_config.NUM_DECONV_FILTERS
        output_size = 2048
        for i in range(self.volume_config.NUM_DECONV_LAYERS):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=output_size,
                    out_channels=filters,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.volume_config.DECONV_WITH_BIAS))
            layers.append(nn.BatchNorm2d(filters, momentum=self.volume_config.BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            output_size = filters

        self._head = nn.Sequential(*layers, nn.Conv2d(
            in_channels=filters,
            out_channels=num_joints * depth_res,
            kernel_size=self.volume_config.FINAL_CONV_KERNEL,
            stride=1,
            padding=0
        ))

    def forward(self, x):
        cnn_features = self._backbone.forward(x)
        heatmaps = self._head(cnn_features)
        # Check trainable parameters
        pred_jnts = self._softmax_integral(heatmaps)
        return pred_jnts

    # FROM https://github.com/JimmySuen/integral-human-pose/blob/master/pytorch_projects/common_pytorch/common_loss/integral.py
    def _softmax_integral(self, heatmaps):
        # global softmax
        depth, height, width = heatmaps.shape[1:]
        depth = depth // self.num_joints
        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, -1))
        heatmaps = self._softmax(heatmaps)

        # integrate heatmap into joint location
        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, depth, height, width))

        x = heatmaps.sum(dim=2)
        x = x.sum(dim=2)
        y = heatmaps.sum(dim=2)
        y = y.sum(dim=3)
        z = heatmaps.sum(dim=3)
        z = z.sum(dim=3)

        x = x * self._range
        y = y * self._range
        z = z * self._range

        x = x.sum(dim=2, keepdim=True)
        y = y.sum(dim=2, keepdim=True)
        z = z.sum(dim=2, keepdim=True)

        x = x / float(width) - 0.5
        y = y / float(height) - 0.5
        z = z / float(depth) - 0.5
        pred_jnts = torch.cat((x, y, z), dim=2)
        return pred_jnts

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

    # if is_train and cfg.MODEL.INIT_WEIGHTS:
    #    model.init_weights(cfg.MODEL.PRETRAINED)

    return model
