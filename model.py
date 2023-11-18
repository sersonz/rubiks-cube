import os
import numpy as np
import pycuber as pc
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import RMSprop
from utils import ACTIONS, get_state, is_solved

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REUSE_DATA = False

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
        ])

        self.fc_policy = nn.Linear(2048, 512)
        self.policy_head = nn.Linear(512, 12)

        self.fc_value = nn.Linear(2048, 512)
        self.value_head = nn.Linear(512, 1)

        self.to(device)

    def forward(self, x):
        assert type(x) == torch.Tensor, "Input must be a tensor"
        x = x.view(-1, 20 * 24)  # flatten input

        for layer in self.shared_layers:
            x = F.elu(layer(x))

        value = F.elu(self.fc_value(x))
        value = self.value_head(value)
        #value = torch.tanh(value)  # [-1, 1]

        policy = F.elu(self.fc_policy(x))
        policy = self.policy_head(policy)
        policy = F.softmax(policy, dim=1)  # prob dist

        return value, policy


def gen_data(adinet, k=5, l=100):
    # Generate training samples

    X = torch.empty((0, 20 * 24), dtype=torch.float32, device=device)
    Y_policy = torch.empty(0, dtype=torch.long, device=device)
    Y_value = torch.empty(0, dtype=torch.float32, device=device)
    with tqdm(total=k * l, desc="Generating data") as pbar:
        for _ in range(l):
            steps = np.random.choice(ACTIONS, k, replace=True)

            # each scramble gives k cubes
            # D(xi) = idx+1
            xis = []
            yvs = []
            yps = []
            cube = pc.Cube()

            for step in steps:
                cube(str(step))
                state = get_state(cube).to(device)  # 480x1
                xis.append(state.view(1, -1))  # 1x480

                # depth-1 BFS for the best action
                yv = float("-inf")
                yp = -1
                for aidx, a in enumerate(ACTIONS):
                    _cube = cube.copy()
                    _cube(a)
                    state = get_state(_cube).to(device)
                    vi, pi = adinet(state)
                    vi += 1 if is_solved(_cube) else -1
                    if vi > yv:
                        yv = vi
                        yp = aidx  # argmax_a(R + vi)
                yvs.append(yv)
                yps.append(yp)
                pbar.update()

            X = torch.cat((X, torch.cat(xis)), 0)
            Y_value = torch.cat(
                (Y_value, torch.tensor(yvs, dtype=torch.float32, device=device)), 0
            )
            Y_policy = torch.cat(
                (Y_policy, torch.tensor(yps, dtype=torch.long, device=device)), 0
            )

    if REUSE_DATA:
        print("Saving data")
        torch.save(X, "X.pt")
        torch.save(Y_value, "Y_value.pt")
        torch.save(Y_policy, "Y_policy.pt")

    return X, Y_value, Y_policy


def train(k=5, l=100, batch_size=32, epochs=10, lr=3e-4, path="./model.pth"):
    """
    Generate training samples by starting with a solved cube,
    scrambled k times (gives a sequence of k cubes)
    repeated l times for N=k*l training samples
    """
    print("Training")
    print(f"{k=}, {l=} -> N={k*l}")

    adinet = ADINet()

    # Loss functions
    value_criterion = nn.MSELoss(reduction="none")
    policy_criterion = nn.CrossEntropyLoss(reduction="none")

    D_xi = torch.cat([torch.arange(1, k + 1)
                      for _ in range(l)]).type(torch.float32).to(device)
    weights = torch.reciprocal(D_xi)

    optimizer = RMSprop(adinet.parameters(), lr=lr)

    if REUSE_DATA and os.path.exists("X.pt"):
        print("Loading data from file")
        print("set REUSE_DATA=False in script or delete X.pt to generate new data instead")
        X = torch.load("X.pt")
        Y_value = torch.load("Y_value.pt")
        Y_policy = torch.load("Y_policy.pt")

        if (X.shape[0] != k*l):
            raise ValueError(f"Loaded data shape does not match params in script (k and l). Loaded {X.shape[0]} samples, expected k*l={k}*{l}={k*l} samples.")

    else:
        X, Y_value, Y_policy = gen_data(adinet, k, l)

    dataset = TensorDataset(X, Y_value, Y_policy, weights)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in pbar:
                x_batch, yv_batch, yp_batch, w_batch = batch

                # Forward pass
                values, policies = adinet(x_batch)
                values = values.squeeze(1)

                # Calculate loss
                loss_value = value_criterion(values, yv_batch)
                loss_value = (loss_value * w_batch).mean()

                loss_policy = policy_criterion(policies, yp_batch)
                loss_policy = (loss_policy * w_batch).mean()

                loss = loss_value + loss_policy

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())

            # print(f"Epoch {epoch+1}/{epochs}: {loss.item()}")  # type: ignore

    # Save the final model
    torch.save(adinet.state_dict(), path)

    return adinet


if __name__ == "__main__":
    REUSE_DATA = True
    train(k=5, l=100, batch_size=8, lr=1e-5, epochs=100)
