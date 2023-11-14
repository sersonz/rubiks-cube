import random
import tqdm
import numpy as np
import pycuber as pc
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ACTIONS, get_state


class ADINet(nn.Module):
    """Architecture for fθ:
            20x24
              |
            4096
              |
            2048
           /    \
        512     512
         |       |
        12       1

    Each layer is fully connected.
    Use elu activation on all layers except for the outputs.
    Combined value and policy network."""
    def __init__(self):
        super(ADINet, self).__init__()

        self.shared_layers = nn.ModuleList([
            nn.Linear(20 * 24, 4096),
            nn.Linear(4096, 2048),
            nn.Linear(2048, 512),
        ])

        self.fc_policy = nn.Linear(512, 512)
        self.policy_head = nn.Linear(512, 12)

        self.fc_value = nn.Linear(512, 512)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(-1, 20 * 24)  # flatten input

        for layer in self.shared_layers:
            x = F.elu(layer(x))

        value = F.elu(self.fc_value(x))
        value = self.value_head(value)
        value = torch.tanh(value)  # [-1, 1]

        policy = F.elu(self.fc_policy(x))
        policy = self.policy_head(policy)
        policy = F.softmax(policy, dim=1)  # prob dist

        return value, policy



def train(k=5, l=100):
    """
    Generate training samples by starting with a solved cube,
    scrambled k times (gives a sequence of k cubes)
    repeated l times for N=k*l training samples
    """

    print("Training")
    print(f"{k=}, {l=} -> N={k*l}")

    adinet = ADINet()

    # Generate training samples
    X = []
    x_steps = []
    for _ in range(l):
        steps = np.random.choice(ACTIONS, k, replace=True)
        x_steps.append(steps)

        # each scramble gives k cubes
        # D(xi) = idx+1
        xis = []
        cube = pc.Cube()
        for step in steps:
            cube(str(step))
            xis.append(cube.copy())

        X.append(xis)

    print(list(zip(x_steps, X)))
    print(len(sum(X, [])))
    # print(x_steps)

    for x in X:
        for i, xi in enumerate(x):
            Dxi = i + 1
            yv = 0
            yp = ""

            for ai,a in enumerate(ACTIONS):




    #     for each training sample:
    #         Perform a depth-1 breadth-first search (BFS)
    #         for each child state in BFS:
    #             Evaluate and store its value
    #         Determine the value target as the maximum value among child states
    #         Determine the policy target as the action leading to the maximum value
    #     Train the neural network using RMSProp optimizer
    #     Loss function = Mean squared error (for value) + Softmax cross-entropy (for policy)
    # return trained neural network
    pass


def solve():
    """
Initialize MCTS tree with initial_state as root
while solution not found and within computational limits:
    current_state = root of the MCTS tree
    while current_state is not terminal:
        if current_state is not fully expanded:
            Expand current_state by adding one or more child states
        Use trained_network to evaluate child states
        Select next state using tree policy (balance exploration and exploitation)
        current_state = next state
    Backpropagate the result (win/loss) up the tree
return solution path from root to terminal state
"""
    pass


if __name__ == "__main__":
    # adinet = ADINet()
    train(k=3, l=3)
