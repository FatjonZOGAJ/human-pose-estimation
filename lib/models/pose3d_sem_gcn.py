import numpy as np
import torch

from lib.models.sem_gcn import SemGCN
from lib.utils.sem_gcn_utils.graph_utils import adj_mx_from_skeleton
from lib.utils.sem_gcn_utils.h36m_dataset import Human36mDataset, H36M_NAMES, h36m_skeleton_joints_group
from lib.utils.sem_gcn_utils.skeleton import h36m_skeleton, joints_group


class PoseGCNRegressor(torch.nn.Module):
    def __init__(self, cfg, use_2d_gt=True, input_joint_dim=2, remove_static_joints=True, **kwargs):
        self.input_joint_dim = input_joint_dim
        self.use_2d_gt = use_2d_gt
        super(PoseGCNRegressor, self).__init__()
        self.skeleton = h36m_skeleton
        if remove_static_joints:
            # Bring the skeleton to 16 joints instead of the original 32
            joints = []
            for i, x in enumerate(H36M_NAMES):
                # TODO: do not remove Neck/Nose and instead use 17 inputs and outputs
                if x == '' or x == 'Neck/Nose':  # Remove 'Nose' to make SH and H36M 2D poses have the same dimension
                    joints.append(i)
            self.skeleton.remove_joints(joints)  # adapted from h36m_dataset and mocap_dataset

            # Rewire shoulders to the correct parents
            self.skeleton._parents[10] = 8
            self.skeleton._parents[13] = 8

            # Set joints group
            self.skeleton._joints_group = h36m_skeleton_joints_group
        adj = adj_mx_from_skeleton(self.skeleton)
        # TODO change dropout and num_layers later
        self.model_pos = SemGCN(adj, cfg.MODEL.SEM_GCN.HID_DIM, num_layers=cfg.MODEL.SEM_GCN.NUM_LAYERS, p_dropout=None,
                                nodes_group=joints_group)

    def forward(self, x):
        if self.use_2d_gt:
            # Drop third dimension z and input is label
            x = x[:, np.mod(np.arange(x.shape[1]), 3) != 2]
        x = x.reshape(x.shape[0], -1, self.input_joint_dim)  # -1 takes care of 16 and 17 joints case
        x = self.model_pos(x)
        return x


def get_pose_net(cfg, **kwargs):
    model = PoseGCNRegressor(cfg, **kwargs)
    return model
