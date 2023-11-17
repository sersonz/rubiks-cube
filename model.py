import random
import tqdm
import numpy as np
import pycuber as pc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import RMSprop
from utils import ACTIONS, get_state, is_solved

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.optimizer = RMSprop(self.parameters())

        self.to(device)

    def forward(self, x):
        assert type(x) == torch.Tensor, "Input must be a tensor"
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


def gen_data(adinet, k=5, l=100):
    # Generate training samples
    X = []
    x_steps = []
    Y_policy = []
    Y_value = []
    for _ in range(l):
        steps = np.random.choice(ACTIONS, k, replace=True)
        x_steps.append(steps)

        # each scramble gives k cubes
        # D(xi) = idx+1
        xis = []
        yvs = []
        yps = []
        cube = pc.Cube()
        for step in steps:
            cube(str(step))
            # xis.append(cube.copy())
            xis.append(get_state(cube))

            yv = float("-inf")
            yp = -1
            # depth-1 BFS
            for aidx, a in enumerate(ACTIONS):
                _cube = cube.copy()
                _cube(a)
                vi, pi = adinet(get_state(_cube))
                vi += 1 if is_solved(_cube) else -1
                if vi > yv:
                    yv = vi
                    yp = aidx  # argmax_a(R + vi)
            yvs.append(yv)
            yps.append(yp)

        X.append(xis)
        Y_value.append(yvs)
        Y_policy.append(yps)

    # print(list(zip(x_steps, X)))
    # print(list(zip(Y_value, Y_policy)))
    print(list(x_steps))
    # print((X[0],))
    print([[yv.data for yv in yvs] for yvs in Y_value])
    print([[ACTIONS[i] for i in yps] for yps in Y_policy])

    # return X, Y_value, Y_policy
    return sum(X, []), sum(Y_value, []), sum(Y_policy, [])


def train(k=5, l=100, epochs=10, path="./model.pth", batch_size=32):
    """
    Generate training samples by starting with a solved cube,
    scrambled k times (gives a sequence of k cubes)
    repeated l times for N=k*l training samples
    """
    print("Training")
    print(f"{k=}, {l=} -> N={k*l}")

    adinet = ADINet()

    # Loss functions
    criterion_value = nn.MSELoss()
    criterion_policy = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        X, Y_value, Y_policy = gen_data(adinet, k, l)
        print(type(X))
        print(type(X[0]))

        # Convert to tensors and create dataset
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        Y_value_tensor = torch.tensor(Y_value, dtype=torch.float32).to(device)
        Y_policy_tensor = torch.tensor(Y_policy, dtype=torch.long).to(device)

        dataset = TensorDataset(X_tensor, Y_value_tensor, Y_policy_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch in dataloader:
            x_batch, yv_batch, yp_batch = batch

            # Forward pass
            values, policies = adinet(x_batch)

            # Calculate loss
            loss_value = criterion_value(values, yv_batch)
            loss_policy = criterion_policy(policies, yp_batch)
            loss = loss_value + loss_policy

            # Backpropagation
            adinet.optimizer.zero_grad()
            loss.backward()
            adinet.optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}: {loss.item()}")  # type: ignore

    # Save the final model
    torch.save(adinet.state_dict(), path)

    return adinet

if __name__ == "__main__":
    train(k=2, l=3, batch_size=32)
