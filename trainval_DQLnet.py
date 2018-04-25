from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
import logging
import time

from model.Reinforcement.utils import init_log
from model.Reinforcement.Player import Player

def load_config(config_path):
    assert(os.path.exists(config_path))
    cfg = json.load(open(config_path, 'r'))
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--config', default='',
                        help='where you put config file')

    args = parser.parse_args()
    return args

def main():
    init_log('global', logging.INFO)
    logger = logging.getLogger('global')

    args = parse_args()
    cfg = load_config(args.config)

    logger.info("logger {}".format(cfg))

    player = Player(cfg)
    player.train()

if __name__ == "__main__":
    main()