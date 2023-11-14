import pycuber
import numpy as np
import argparse
import random
import torch
from utils import ACTIONS, get_state
from model import ADINet, train

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="./model.pth",
    help="The path to save/load the ADINet",
)
parser.add_argument(
    "-t",
    "--train",
    action="store_true",
    help="Whether to train the ADINet",
)
parser.add_argument(
    "-k",
    type=int,
    help="Number of random moves to apply to each cube when generating training samples",
)
parser.add_argument(
    "-l",
    type=int,
    help="Number of cubes to scramble when generating training samples",
)
parser.add_argument("-c", "--count", type=int, default=1)
args = parser.parse_args()

# Init + train ADINet

if args.train:
    adinet = train(args.k, args.l, path=args.model)
    adinet.save(args.model)

adinet = torch.load(args.model)

# gen some test cubes
# solve with ADINet + MCTS solver
# score/eval
