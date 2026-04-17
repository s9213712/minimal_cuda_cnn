#!/usr/bin/env python3
"""PyTorch ResNet-20 on CIFAR-10 for reference"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle, time

workspace = "NN/minimal_cuda_cnn/data/cifar-10-batches-py"

with open(f"{workspace}/data_batch_1", "rb") as f:
    b = pickle.load(f, encoding="bytes")
    x_all = b[b"data"].astype(np.float32) / 255.0
    x_all = x_all.reshape(-1, 3, 32, 32)
    y_all = np.array(b[b"labels"])

print(f"Data: {x_all.shape}")

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.nn.functional.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

model = ResNet(BasicBlock, [3, 3, 3])  # ResNet-20
print(f"ResNet-20 params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = nn.CrossEntropyLoss()

BATCH = 128
EPOCHS = 9
NBATCHES = x_all.shape[0] // BATCH

for epoch in range(EPOCHS):
    t0 = time.time()
    total_loss = 0.0; correct = 0
    indices = np.random.permutation(x_all.shape[0])
    model.train()
    
    for batch_idx in range(NBATCHES):
        idx_s = batch_idx * BATCH; idx_e = idx_s + BATCH
        x = torch.from_numpy(x_all[indices[idx_s:idx_e]])
        y = torch.from_numpy(y_all[indices[idx_s:idx_e]]).long()
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
    
    scheduler.step()
    acc = correct / NBATCHES / BATCH * 100
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss={total_loss/NBATCHES:.4f}, Acc={acc:.2f}%, LR={lr:.4f}, Time={time.time()-t0:.1f}s")

print("\nPyTorch ResNet-20 done!")
