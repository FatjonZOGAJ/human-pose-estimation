import torch
import torch.nn as nn


class DAE(nn.Module):
    def __init__(self, cfg):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(cfg.MODEL.NUM_JOINTS * 3, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, cfg.MODEL.NUM_JOINTS * 3),
        )

    def forward(self, x):
        batch, joints, dims = x.shape
        x = x.reshape([batch, joints * dims])
        x = self.encoder(x)

        x = self.decoder(x)
        x = x.reshape([batch, joints, dims])

        return x
