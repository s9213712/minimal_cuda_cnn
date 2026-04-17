import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_alexnet_pytorch():
    class AlexNet(nn.Module):
        def __init__(self):
            super(AlexNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 192, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(192, 384, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(256 * 2 * 2, 4096), nn.ReLU(),
                nn.Linear(4096, 4096), nn.ReLU(),
                nn.Linear(4096, 10),
            )
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    return AlexNet()

def train_and_eval():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    model = get_alexnet_pytorch()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("Starting PyTorch Training (CPU Mode)...")
    model.train()
    for epoch in range(1): # Single epoch for speed of verification
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs, labels
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0: print(f"Batch {i}, Loss: {loss.item():.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs, labels
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"PyTorch Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train_and_eval()
