import torch
import torch.nn as nn


class DAE(nn.Module):
    def __init__(self, cfg):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(cfg.MODEL.NUM_JOINTS * 3, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, cfg.MODEL.NUM_JOINTS * 3),
            nn.Tanh()
        )

    def forward(self, x):
        batch, joints, dims = x.shape
        x = x.reshape([batch, joints * dims])
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape([batch, joints, dims])

        return x
