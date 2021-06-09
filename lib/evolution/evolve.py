import torch
import numpy as np

import torch.utils.data

from lib.utils.utils import calc_kpt_bound
from scripts.train import parse_args, reset_config
import lib.dataset as dataset
from lib.core.config import config, gen_config
from lib.evolution.genetic import evolution
from lib.evolution.parameter import parse_arg_evolution
from lib.utils.img_utils import gen_trans_from_patch_cv, trans_point2d

torch.manual_seed(3)


def normalize(population, db):
    for i in range(len(population)):
        population[i] = population[i].reshape((17, 3))
        c_x = population[i][0, 0]
        c_y = population[i][0, 1]
        u, d, l, r = calc_kpt_bound(population[i], np.ones((17, 3)))
        bb_width = r - l
        bb_height = d - u
        trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, db.patch_width, db.patch_height, 1.0, 0, inv=False)
        for n_jt in range(len(population[i])):
            population[i][n_jt, 0:2] = trans_point2d(population[n_jt, 0:2], trans)
            population[i][n_jt, 2] = population[n_jt, 2] / db.rect_3d_width * db.patch_width

        population[i][:, 0] = population[i][:, 0] / db.patch_width - 0.5
        population[i][:, 1] = population[i][:, 1] / db.patch_height - 0.5
        population[i][:, 2] = population[i][:, 2] / db.patch_width


def normalize2(population):
    population = list(population)
    for i in range(len(population)):
        population[i] = population[i].reshape((17, 3))
        population[i] = population[i] - population[i][0]
        population[i] = (population[i] - population[i].mean(axis=0)) / population[i].std(axis=0)
        population[i] = population[i] - population[i][0]
        population[i] = population[i].reshape((-1))

    return np.array(population)


def save_to_csv(population, file_name):
    np.savetxt(file_name, population, delimiter=",")


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
    valid_dataset = ds(
        cfg=config,
        root=config.DATASET.ROOT,
        image_set=config.DATASET.TEST_SET,
        is_train=False
    )

    initial_population = []
    initial_population_val = []
    for db in train_dataset.db:
        initial_population.append(db['joints_3d'].reshape((-1)))
        p = db['joints_3d']
        u, d, l, r = calc_kpt_bound(p, np.ones((17, 3)))
        bb_width = r - l
        bb_height = d - u

    for db in valid_dataset.db:
        initial_population_val.append(db['joints_3d'].reshape((-1)))

    evolved_population = evolution(initial_population, opt, model_file=None)
    print("Evolved train dataset")
    save_to_csv(evolved_population, "evolved2.csv")
    evolved_population = normalize2(evolved_population)
    save_to_csv(evolved_population, "evolved_normalized2.csv")

    evolved_population = evolution(initial_population_val, opt, model_file=None)
    print("Evolved validation dataset")
    save_to_csv(evolved_population, "evolved_val2.csv")
    evolved_population = normalize2(evolved_population)
    save_to_csv(evolved_population, "evolved_normalized_val2.csv")

    # logger.info('=> loading the valid dataset')

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
