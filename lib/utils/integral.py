import torch
from torch.nn import functional as F


# From https://github.com/JimmySuen/integral-human-pose/blob/ad2f875bc05538da3471ef81484e23fad3e9c787/pytorch_projects/common_pytorch/common_loss/integral.py#L28
def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim, z_dim):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, z_dim, y_dim, x_dim))
    x, y, z = [None, None, None]

    if x_dim > 1 and y_dim > 1:
        accu_x = heatmaps.sum(dim=2)
        accu_x = accu_x.sum(dim=2)
        accu_x = accu_x * torch.arange(start=1, end=x_dim + 1, device=[accu_x.device.index][0])
        x = accu_x.sum(dim=2, keepdim=True)

        accu_y = heatmaps.sum(dim=2)
        accu_y = accu_y.sum(dim=3)
        accu_y = accu_y * torch.arange(start=1, end=y_dim + 1, device=[accu_y.device.index][0])
        y = accu_y.sum(dim=2, keepdim=True)

    if z_dim > 1:
        accu_z = heatmaps.sum(dim=3)
        accu_z = accu_z.sum(dim=3)
        accu_z = accu_z * torch.arange(start=1, end=z_dim + 1, device=[accu_z.device.index][0])
        z = accu_z.sum(dim=2, keepdim=True)

    return x, y, z


# preds must be of form [batch,n_joints,depth,height,width]
def softmax_integral_tensor(preds, num_joints, output, hm_width, hm_height, hm_depth):
    # integrate heatmap into joint location
    # global soft max
    preds = preds.reshape((preds.shape[0], num_joints, -1))
    # normalize heatmap of each joint to 1
    preds = F.softmax(preds, 2)

    if output == 3:
        x, y, z = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
        x = x / float(hm_width) - 0.5
        y = y / float(hm_height) - 0.5
        z = z / float(hm_depth) - 0.5

        # return calculated coordinates in shape [32,17,(x,y,z)]
        preds = torch.cat((x, y, z), dim=2)
    elif output == 2:
        x, y, _ = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
        x = x / float(hm_width) - 0.5
        y = y / float(hm_height) - 0.5
        preds = torch.cat((x, y), dim=2)
    else:
        _, _, z = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
        z = z / float(hm_depth) - 0.5
        preds = z

    return preds
