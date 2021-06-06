import torch

import torch.utils.data
from scripts.train import parse_args, reset_config
import lib.dataset as dataset
from lib.core.config import config, gen_config
from lib.utils.utils import create_logger
from lib.evolution.genetic import evolution
from lib.evolution.parameter import parse_arg_evolution
from lib.utils.img_utils import gen_trans_from_patch_cv, trans_point2d

torch.manual_seed(3)


def normalize(population, db):
    for i in range(len(population)):
        population[i] = population[i].reshape((17, 3))
        c_x = population[i][0, 0]
        c_y = population[i][0, 1]
        bb_width = 1
        bb_height = 1
        trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, db.patch_width, db.patch_height, 1.0, 0, inv=False)
        for n_jt in range(len(population[i])):
            population[i][n_jt, 0:2] = trans_point2d(population[n_jt, 0:2], trans)
            population[i][n_jt, 2] = population[n_jt, 2] / db.rect_3d_width * db.patch_width

        population[i][:, 0] = population[i][:, 0] / db.patch_width - 0.5
        population[i][:, 1] = population[i][:, 1] / db.patch_height - 0.5
        population[i][:, 2] = population[i][:, 2] / db.patch_width


if __name__ == "__main__":
    args = parse_args()
    reset_config(config, args)
    opt = parse_arg_evolution()

    ds = dataset.h36m

    # Data loading code
    train_dataset = ds(
        cfg=config,
        root=config.DATASET.ROOT,
        image_set=config.DATASET.TRAIN_SET,
        is_train=True
    )

    initial_population = []
    for db in train_dataset.db:
        initial_population.append(db['joints_3d'].reshape((-1)))

    evolved_population = evolution(initial_population,
                                   opt,
                                   model_file=None
                                   )

    evolved_population = normalize(evolved_population, train_dataset.db)
    # logger.info('=> loading the valid dataset')
    # valid_dataset = ds(
    #     cfg=config,
    #     root=config.DATASET.ROOT,
    #     image_set=config.DATASET.TEST_SET,
    #     is_train=False
    # )
    #
    # gpus = [int(i) for i in config.GPUS.split(',')]
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
    #     shuffle=config.TRAIN.SHUFFLE,
    #     num_workers=config.WORKERS,
    #     pin_memory=True
    # )
    # valid_loader = torch.utils.data.DataLoader(
    #     valid_dataset,
    #     batch_size=config.TEST.BATCH_SIZE * len(gpus),
    #     shuffle=False,
    #     num_workers=config.WORKERS,
    #     pin_memory=True
    # )
