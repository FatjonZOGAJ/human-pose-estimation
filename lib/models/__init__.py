import lib.models.pose3d_alexnet_reg
from lib.models import pose3d_vol, gcn_end_to_end

nets = {'pose3d_alexnet_reg': pose3d_alexnet_reg,
        'pose3d_vol': pose3d_vol,
        'sem_gcn': gcn_end_to_end
        }
