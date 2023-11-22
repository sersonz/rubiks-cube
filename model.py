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
    def __init__(self, dropout=False):
        super(ADINet, self).__init__()

        self.shared_layers = nn.ModuleList([
            nn.Linear(20 * 24, 4096),
            nn.Linear(4096, 2048),
        ])

        self.policy_layer = nn.Linear(2048, 512)
        self.policy_head = nn.Linear(512, 12)

        self.value_layer = nn.Linear(2048, 512)
        self.value_head = nn.Linear(512, 1)

        self.dropout = dropout

        self.init_weights()
        self.to(device)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        assert type(x) == torch.Tensor, "Input must be a tensor"
        x = x.view(-1, 20 * 24)  # flatten input

        for layer in self.shared_layers:
            x = F.elu(layer(x))
            if self.dropout:
                x = F.dropout(x, p=0.2, training=self.training)

        value = self.value_head(F.elu(self.value_layer(x)))
        policy = self.policy_head(F.elu(self.policy_layer(x)))

        return value, policy


def gen_data(adinet, k, l, modified_target=False):
    # Generate training samples

    X = torch.empty((0, 20 * 24), dtype=torch.float32, device=device)
    Y_policy = torch.empty(0, dtype=torch.long, device=device)
    Y_value = torch.empty(0, dtype=torch.float32, device=device)
    with tqdm(total=k * l, desc="Generating data") as pbar:
        for _ in range(l):
            steps = np.random.choice(ACTIONS, k, replace=True)

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
                    if modified_target:
                        if is_solved(_cube):
                            vi = 0
                        else:
                            vi -= 1
                    else:
                        vi += 1 if is_solved(_cube) else -1
                    if vi > yv:
                        yv = vi
                        yp = aidx  # argmax_a(R + vi)
                yvs.append(yv)
                yps.append(yp)
                pbar.update()

            X = torch.cat((X, torch.cat(xis)), 0)
            Y_value = torch.cat(
                (
                    Y_value,
                    torch.tensor(yvs, dtype=torch.float32, device=device),
                ),
                0,
            )
            Y_policy = torch.cat(
                (
                    Y_policy,
                    torch.tensor(yps, dtype=torch.long, device=device),
                ),
                0,
            )

    return X, Y_value, Y_policy


def train(
    k=5,
    l=100,
    batch_size=32,
    epochs=10,
    iterations=100,
    lr=3e-4,
    path="./model.pth",
    load=False,
    modified_target=False,
    dropout=False,
):
    """
    Generate training samples by starting with a solved cube,
    scrambled k times (gives a sequence of k cubes)
    repeated l times for N=k*l training samples
    """
    print("Training")
    print(f"{k=}, {l=} -> N={k*l}")

    subpath = path[:path.rfind(".pth")]

    # Loss functions
    value_criterion = nn.MSELoss(reduction="none")
    policy_criterion = nn.CrossEntropyLoss(reduction="none")

    D_xi = torch.cat([torch.arange(1, k + 1)
                      for _ in range(l)]).type(torch.float32).to(device)
    loss_weights = torch.reciprocal(D_xi)

    model = ADINet(dropout=dropout)
    optimizer = RMSprop(model.parameters(), lr=lr)
    if load:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        iter = checkpoint["iter"]
        hist = checkpoint["hist"]
    else:
        iter = 0
        hist = []
    model.train()

    for i in range(iter, iter + iterations):
        print(f"\nIteration {i+1}/{iter + iterations}")

        X, Y_value, Y_policy = gen_data(model, k, l, modified_target)
        dataset = TensorDataset(X, Y_value, Y_policy, loss_weights)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                epoch_loss = 0.0
                for batch in pbar:
                    x, y_values, y_policies, batch_loss_weights = batch

                    # Forward pass
                    pred_values, pred_policies = model(x)
                    pred_values = pred_values.squeeze(1)

                    # Calculate loss
                    loss_value = value_criterion(pred_values, y_values)
                    loss_value = (loss_value * batch_loss_weights).mean()

                    loss_policy = policy_criterion(pred_policies, y_policies)
                    loss_policy = (loss_policy * batch_loss_weights).mean()

                    loss = loss_value + loss_policy

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix(loss=loss.item())
                    epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            hist.append(avg_epoch_loss)
            pbar.set_postfix(loss=avg_epoch_loss)

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": i + 1,
            "hist": hist,
        }, f"{subpath}_{i}.pth")

    # Save the final model
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": iter + iterations,
        "hist": hist,
    }, f"{subpath}_FINAL.pth")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k",
        type=int,
        required=True,
        help="Number of random moves to apply to each cube when generating training samples",
    )
    parser.add_argument(
        "-l",
        type=int,
        required=True,
        help="Number of cubes to scramble when generating training samples",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train each iteration",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations to train",
    )
    parser.add_argument(
        "-r",
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./models/model.pth",
        help="The path to save/load the ADINet",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load model path as a training checkpoint and continue training",
    )
    parser.add_argument(
        "--mod_target",
        action="store_true",
        help="Use modified method for target value calculation",
    )
    parser.add_argument(
        "--dropout",
        action="store_true",
        help="Use dropout in model training",
    )
    args = parser.parse_args()

    train(
        k=args.k,
        l=args.l,
        batch_size=args.batch_size,
        epochs=args.epochs,
        iterations=args.iterations,
        lr=args.lr,
        path=args.model,
        load=args.load,
        modified_target=args.mod_target,
        dropout=args.dropout,
    )
