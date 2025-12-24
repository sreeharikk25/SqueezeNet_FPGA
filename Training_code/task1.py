# -*- coding: utf-8 -*-
"""
EE 511 - Task 1: SqueezeNet v1.0 FP32 Training (No BatchNorm)
=============================================================
- Trains SqueezeNet on CIFAR-10
- Saves best checkpoint
- Generates loss and accuracy plots
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {DEVICE}")


# =============================================================================
# Fire Module (No BatchNorm)
# =============================================================================

class Fire(nn.Module):
    def __init__(self, in_ch, sq_ch, e1_ch, e3_ch):
        super().__init__()
        self.squeeze = nn.Conv2d(in_ch, sq_ch, 1, bias=True)
        self.expand1x1 = nn.Conv2d(sq_ch, e1_ch, 1, bias=True)
        self.expand3x3 = nn.Conv2d(sq_ch, e3_ch, 3, padding=1, bias=True)

        for m in [self.squeeze, self.expand1x1, self.expand3x3]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.squeeze(x))
        o1 = self.expand1x1(x)
        o3 = self.expand3x3(x)
        return F.relu(torch.cat([o1, o3], 1))


# =============================================================================
# SqueezeNet v1.0 (No BatchNorm)
# =============================================================================

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 7, stride=2, padding=3, bias=True)

        self.fire2 = Fire(96, 16, 64, 64)
        self.fire3 = Fire(128, 16, 64, 64)
        self.fire4 = Fire(128, 32, 128, 128)
        self.fire5 = Fire(256, 32, 128, 128)
        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)
        self.fire8 = Fire(384, 64, 256, 256)
        self.fire9 = Fire(512, 64, 256, 256)

        self.pool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.dropout = nn.Dropout(0.5)
        self.conv10 = nn.Conv2d(512, num_classes, 1, bias=True)
        self.avg = nn.AdaptiveAvgPool2d(1)

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv10.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv10.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.pool(x)

        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.pool(x)

        x = self.fire9(x)
        x = self.dropout(x)
        x = F.relu(self.conv10(x))
        x = self.avg(x)

        return torch.flatten(x, 1)


# =============================================================================
# Plot Training Curves
# =============================================================================

def plot_training_curves(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    test_acc = [h['test_acc'] for h in history]

    # Combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, 'b-', linewidth=2, marker='o', markersize=4, label='Train')
    ax2.plot(epochs, test_acc, 'r-', linewidth=2, marker='s', markersize=4, label='Test')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'training_curves.pdf'), bbox_inches='tight')
    plt.show()

    print(f"\nSaved: {output_dir}/training_curves.png")
    print(f"Saved: {output_dir}/training_curves.pdf")


# =============================================================================
# Test Function
# =============================================================================

def test(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


# =============================================================================
# Data Loaders
# =============================================================================

transform_train = T.Compose([
    T.Resize(224),
    T.RandomHorizontalFlip(),
    T.RandomCrop(224, padding=4),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainloader = DataLoader(
    torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform_train),
    batch_size=128, shuffle=True, num_workers=2, pin_memory=True
)
testloader = DataLoader(
    torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform_test),
    batch_size=128, shuffle=False, num_workers=2, pin_memory=True
)


# =============================================================================
# Main Training
# =============================================================================

def main():
    print("\n" + "="*70)
    print("EE 511 - Task 1: SqueezeNet v1.0 FP32 Training")
    print("="*70)

    os.makedirs("checkpoints", exist_ok=True)

    model = SqueezeNet(num_classes=10).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: SqueezeNet v1.0 (No BatchNorm)")
    print(f"Parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    print(f"\nHyperparameters:")
    print(f"  Optimizer: SGD (lr=0.01, momentum=0.9, weight_decay=1e-4)")
    print(f"  Scheduler: CosineAnnealing (T_max=100)")
    print(f"  Epochs: 100")

    history = []
    best_acc = 0

    print("\n" + "-"*70)
    print(f"{'Epoch':<7} {'Loss':<10} {'Train':<10} {'Test':<10} {'Status'}")
    print("-"*70)

    for epoch in range(1, 101):
        # Train
        model.train()
        running_loss = correct = total = 0

        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        scheduler.step()

        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * correct / total
        test_acc = test(model, testloader)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
        })

        status = ""
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'epoch': epoch,
            }, "checkpoints/task1_best.pth")
            status = f"★ Best: {best_acc:.2f}%"

        print(f"{epoch:<7} {train_loss:<10.4f} {train_acc:<10.2f} {test_acc:<10.2f} {status}")

    # Results
    print("\n" + "="*70)
    print(f"TASK 1 COMPLETE")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print("="*70)

    # Plot
    plot_training_curves(history, "checkpoints")

    print(f"\nSaved: checkpoints/task1_best.pth")


if __name__ == '__main__':
    main()
