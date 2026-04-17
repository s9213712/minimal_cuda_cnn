#!/usr/bin/env python3
"""PyTorch version of same architecture for comparison"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import pickle, time

workspace = "NN/minimal_cuda_cnn/data/cifar-10-batches-py"

# Load CIFAR
with open(f"{workspace}/data_batch_1", "rb") as f:
    b = pickle.load(f, encoding="bytes")
    x_all = b[b"data"].astype(np.float32) / 255.0
    x_all = x_all.reshape(-1, 3, 32, 32)
    y_all = np.array(b[b"labels"])

print(f"Data: {x_all.shape}")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=0)   # 3→32, 3x3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=0)  # 32→64, 3x3
        self.fc = nn.Linear(64 * 6 * 6, 10)
        self.relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))       # 32x30x30
        x = torch.max_pool2d(x, 2)         # 32x15x15
        x = self.relu(self.conv2(x))       # 64x13x13
        x = torch.max_pool2d(x, 2)         # 64x6x6
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = Net()
optimizer = optim.SGD([
    {'params': model.conv1.parameters(), 'lr': 0.001},
    {'params': model.conv2.parameters(), 'lr': 0.01},
    {'params': model.fc.parameters(), 'lr': 0.01},
])
criterion = nn.CrossEntropyLoss()

BATCH = 64
EPOCHS = 9  # same as neat-kelp
NBATCHES = x_all.shape[0] // BATCH

for epoch in range(EPOCHS):
    t0 = time.time()
    total_loss = 0.0; correct = 0
    indices = np.random.permutation(x_all.shape[0])
    
    for batch_idx in range(NBATCHES):
        idx_s = batch_idx * BATCH; idx_e = idx_s + BATCH
        x = torch.from_numpy(x_all[indices[idx_s:idx_e]])
        y = torch.from_numpy(y_all[indices[idx_s:idx_e]]).long()
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        for p in model.parameters():
            p.grad.data.clamp_(-1.0, 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
    
    acc = correct / NBATCHES / BATCH * 100
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss={total_loss/NBATCHES:.4f}, Acc={acc:.2f}%, Time={time.time()-t0:.1f}s")

print("\nPyTorch comparison done!")
