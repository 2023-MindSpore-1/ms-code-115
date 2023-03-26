import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))

import argparse
from utils.util import read_yaml
from engine.engine import Engine
import mindspore as ms


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-c', '--config', type=str)

    args = parse.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = read_yaml(args.config)
    engine = Engine(cfg)
    engine.pretrain()
