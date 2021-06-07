import numpy as np
import torch

from lib.models.sem_gcn import SemGCN
from lib.utils.sem_gcn_utils.graph_utils import adj_mx_from_skeleton
from lib.utils.sem_gcn_utils.h36m_dataset import h36m_skeleton, H36M_NAMES, h36m_skeleton_joints_group


# drop_joint_idx corresponds to H36M_NAMES indices starting with 0 (aka [0, 16])
class PoseGCNRegressor(torch.nn.Module):
    def __init__(self, cfg, use_gt_as_input=True, use_2d_input=True, input_joint_dim=2, remove_static_joints=True,
                 drop_joint_idx=-1, move_neck_nose_to_end=True, **kwargs):
        super(PoseGCNRegressor, self).__init__()
        self.input_joint_dim = input_joint_dim
        self.drop_joint_idx = drop_joint_idx
        self.use_gt_as_input = use_gt_as_input
        self.use_2d_input = use_2d_input
        self.move_neck_nose_to_end = move_neck_nose_to_end

        self.skeleton = h36m_skeleton
        if remove_static_joints:
            # Bring the skeleton to 17 joints instead of the original 32
            joints = []
            for i, x in enumerate(H36M_NAMES):
                if x == '':  # or x == 'Neck/Nose':  # Remove 'Nose' to make SH and H36M 2D poses have the same dimension
                    joints.append(i)
            self.skeleton.remove_joints(joints)  # adapted from h36m_dataset and mocap_dataset

            # Rewire shoulders to the correct parents
            self.skeleton._parents[10] = 8
            self.skeleton._parents[13] = 8

            # Set joints group
            self.skeleton._joints_group = h36m_skeleton_joints_group
        adj = adj_mx_from_skeleton(self.skeleton)
        # TODO play around with dropout and num_layers later
        self.model_pos = SemGCN(adj, cfg.MODEL.SEM_GCN.HID_DIM, num_layers=cfg.MODEL.SEM_GCN.NUM_LAYERS,
                                p_dropout=cfg.MODEL.SEM_GCN.DROPOUT, nodes_group=h36m_skeleton_joints_group)

    def forward(self, x):
        # Drop third dimension z
        if self.use_2d_input:
            if len(x.shape) == 3:
                x = x[:, :, :-1]  # (batch_size, 17, 2)
            else:
                x = x[:, np.mod(np.arange(x.shape[1]), 3) != 2]
        x = x.reshape(x.shape[0], -1, self.input_joint_dim)  # -1 takes care of 16 and 17 joints case
        # move Neck/Nose to end as we add it that way to SemGCN
        if not self.use_gt_as_input:        # we reorder gt labels anyway -> these are now input
            if x.shape[1] == 17 and self.move_neck_nose_to_end:
                x = torch.cat([x[:, :9, :], x[:, 10:, :], x[:, 9:10, :]], axis=1)
        x = self.model_pos(x)
        return x


def get_pose_net(cfg, **kwargs):
    model = PoseGCNRegressor(cfg, **kwargs)
    return model
