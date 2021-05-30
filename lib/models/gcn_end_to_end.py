import torch
import lib.models as models

from lib.models.sem_gcn import SemGCN
from lib.utils.graph_utils import adj_mx_from_skeleton
from lib.utils.skeleton import h36m_skeleton, joints_group


class GraphModel(torch.nn.Module):
    def __init__(self, cfg):
        super(GraphModel, self).__init__()
        # TODO add this parameters to config file
        self.model_2d = models.nets['pose3d_vol'].get_pose_net(cfg, True and cfg.MODEL.INIT_WEIGHTS)
        checkpoint = torch.load(cfg.MODEL.CKPT_2D)
        self.model_2d.load_state_dict(checkpoint)
        GraphModel.freeze_model(self.model_2d)

        adj = adj_mx_from_skeleton(h36m_skeleton)
        self. model_pos = SemGCN(adj, cfg.hid_dim, num_layers=4, p_dropout=None,  # TODO change dropout and num_layers later
                           nodes_group=joints_group)

    def forward(self, x):
        x = self.model_2d(x)  # predict 2d joints
        x = x[:, :, :-1]  # (batch_size, 17, 2)
        x = self.gcn(x)
        return x

    @staticmethod
    def freeze_model(model):
        for param in model.parameters():
            param.requires_grad = False


def get_pose_net(cfg, **kwargs):
    model = GraphModel(cfg)
    return model