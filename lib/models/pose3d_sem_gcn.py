import numpy as np
import torch

from lib.models.sem_gcn import SemGCN
from lib.utils.sem_gcn_utils.graph_utils import adj_mx_from_skeleton
from lib.utils.sem_gcn_utils.skeleton import h36m_skeleton, joints_group


# TODO: add to other models as well so that they work with train/validate step
class GroundTruthTrainable(object):
    def __init__(self, use_2d_gt=False):
        self.use_2d_gt = use_2d_gt

    def forward(self, y):
        if self.use_2d_gt:
            # Dropping third dimension z
            return y[:, np.mod(np.arange(y.shape[1]), 3) != 2]
        else:
            return y


class PoseGCNRegressor(torch.nn.Module, GroundTruthTrainable):
    def __init__(self, cfg, use_2d_gt=True, input_joint_dim=2, **kwargs):
        self.input_joint_dim = input_joint_dim
        super(PoseGCNRegressor, self).__init__()
        GroundTruthTrainable.__init__(self, use_2d_gt=use_2d_gt)
        adj = adj_mx_from_skeleton(h36m_skeleton)
        # TODO change dropout and num_layers later
        self.model_pos = SemGCN(adj, cfg.MODEL.SEM_GCN.HID_DIM, num_layers=cfg.MODEL.SEM_GCN.NUM_LAYERS, p_dropout=None, nodes_group=joints_group)

    def forward(self, x):
        x = GroundTruthTrainable.forward(self, x)
        x = x.reshape(x.shape[0], -1, self.input_joint_dim)     # -1 takes care of 16 and 17 joints case
        x = self.model_pos(x)
        return x


def get_pose_net(cfg, **kwargs):
    model = PoseGCNRegressor(cfg, **kwargs)
    return model
