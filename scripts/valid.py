import argparse
import os
import pprint
import _init_paths

import torch
import torch.nn.parallel
import torch.utils.data
from tensorboardX import SummaryWriter

from lib.core.config import config
from lib.core.config import update_config
from lib.core.function import validate, evaluate
from lib.utils.utils import create_logger

import lib.dataset as dataset
import lib.models as models

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    tb_log_dir = os.path.join(final_output_dir, 'logs')

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'valid_global_steps': 0,
    }

    model = models.pose3d_alexnet_reg.get_pose_net(config, is_train=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # Load model
    model_file = os.path.join(final_output_dir, 'model_best.pth.tar')
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint)
    logger.info('=> resume from pretrained model {}'.format(model_file))

    ds = eval('dataset.' + config.DATASET.DATASET)

    # Data loading code
    valid_dataset = ds(
        cfg=config,
        root=config.DATASET.ROOT,
        image_set=config.DATASET.TEST_SET,
        is_train=False
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    preds_in_patch_with_score = validate(valid_loader, model)
    acc = evaluate(0, preds_in_patch_with_score, valid_loader, final_output_dir, config.DEBUG.DEBUG, writer_dict)

if __name__ == '__main__':
    os.chdir('..')
    main()
