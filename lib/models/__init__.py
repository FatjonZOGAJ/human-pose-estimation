import lib.models.pose3d_resnet_vol
import lib.models.pose3d_alexnet_reg
from lib.models import pose3d_resnext_reg

nets = {'pose3d_alexnet_reg': pose3d_alexnet_reg,
        'pose3d_resnext_reg': pose3d_resnext_reg,
        'pose3d_resnet_vol': pose3d_resnet_vol}
