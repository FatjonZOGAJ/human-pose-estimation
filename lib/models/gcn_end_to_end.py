import torch
import lib.models as models
from lib.models.pose3d_sem_gcn import PoseGCNRegressor

from lib.models.sem_gcn import SemGCN
from lib.utils.sem_gcn_utils.graph_utils import adj_mx_from_skeleton
from lib.utils.sem_gcn_utils.h36m_dataset import h36m_skeleton, h36m_skeleton_joints_group


class GraphModel(torch.nn.Module):
    def __init__(self, cfg):
        super(GraphModel, self).__init__()
        self.use_gt_as_input = False
        self.drop_joint_idx = -1
        self.move_neck_nose_to_end = cfg.MODEL.MOVE_NECK_NOSE_TO_END
        self.freeze_backbone_completely = cfg.MODEL.FREEZE_BACKBONE_COMPLETELY

        self.model_2d = models.nets['pose3d_vol'].get_pose_net(cfg, True and cfg.MODEL.INIT_WEIGHTS)
        if cfg.MODEL.CKPT_2D != '':
            checkpoint = torch.load(cfg.MODEL.CKPT_2D)
            # drop 'module.' in key names as we load parts separately
            checkpoint = {k[7:]: v
                          for k, v in checkpoint.items()}
            self.model_2d.load_state_dict(checkpoint)
        # Last Layer of ResNext is not frozen as it was trained that way
        if self.freeze_backbone_completely:
            GraphModel.freeze_model(self.model_2d)

        self.model_pos = PoseGCNRegressor(cfg, use_gt_as_input=False,
                                          input_joint_dim=cfg.MODEL.SEM_GCN.INPUT_JOINT_DIM,
                                          move_neck_nose_to_end=cfg.MODEL.MOVE_NECK_NOSE_TO_END)
        if cfg.MODEL.CKPT_3D != '':
            checkpoint = torch.load(cfg.MODEL.CKPT_3D)
            # drop 'module.' in key names as we load parts separately
            checkpoint = {k[7:]: v
                          for k, v in checkpoint.items()}
            self.model_pos.load_state_dict(checkpoint)
            # when resuming, we load the model from MODEL.TIME_STR again

    def forward(self, x):
        x = self.model_2d(x)  # predict 2d joints
        x = self.model_pos(x)
        return x

    @staticmethod
    def freeze_model(model):
        for param in model.parameters():
            param.requires_grad = False


def get_pose_net(cfg, **kwargs):
    model = GraphModel(cfg)
    return model
