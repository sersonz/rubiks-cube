import torch
import matplotlib.pyplot as plt
import numpy as np

hist = torch.load("models/model.pth")["hist"]

plt.title("Training Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.ylim(0, 1.7)

plt.plot(hist, label="Train loss")
plt.show()

# moving average of 10
avg = []
window = 100
for i in range(len(hist)):
    if i < window:
        avg.append(np.mean(hist[:i+1]))
    else:
        avg.append(np.mean(hist[i-window:i+1]))

plt.title("Training Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.ylim(0, 1.7)

plt.plot(avg, label="Train loss")
plt.show()


