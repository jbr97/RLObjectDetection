from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import argparse
import logging
import time
import torchvision.transforms as transforms

import _init_paths
from model.Reinforcement.utils import init_log
from model.Reinforcement.Player import Player
from datasets.DQL_coco_dataset import COCODataset, COCOTransform
from datasets.DQL_coco_loader import COCODataLoader

def load_config(config_path):
    assert(os.path.exists(config_path))
    cfg = json.load(open(config_path, 'r'))
    return cfg

def main():
    init_log('global', logging.INFO)
    logger = logging.getLogger('global')

    #args = parse_args()
    json_file = sys.argv[1]
    logger.info(json_file)
    cfg = load_config(json_file)
    logger.info("config {}".format(cfg))

    normalize = transforms.Normalize(mean=[0.4485295, 0.4249905, 0.39198247],
                                     std=[0.12032582, 0.12394787, 0.14252729])
    dataset = COCODataset(
        cfg["data_dir"],
        cfg["ann_file"],
        cfg["dt_file"],
        COCOTransform([800], 1200, flip=False),
        normalize_fn=normalize)
    dataloader = COCODataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=6)
    logger.info("Dataset Build Done")

    player = Player(cfg)
    if cfg["mode"] == "train":
        logger.info("Start training!!!")
        player.train(dataloader)
    elif cfg["mode"] == 'val':
        logger.info("-----------Start validate---------")
        player.eval(dataloader)

if __name__ == "__main__":
    main()
