import lib.models.pose3d_alexnet_reg
from lib.models import pose3d_vol, gcn_end_to_end, pose3d_sem_gcn

nets = {'pose3d_alexnet_reg': pose3d_alexnet_reg,
        'pose3d_vol': pose3d_vol,
        'pose3d_sem_gcn': pose3d_sem_gcn,
        'gcn_end_to_end': gcn_end_to_end
        }
