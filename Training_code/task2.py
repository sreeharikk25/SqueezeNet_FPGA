# -*- coding: utf-8 -*-
"""
EE 511 - Task 2: Stable QAT for Full Quantization
Techniques to prevent divergence:
1. Progressive quantization (activations gradually)
2. Lower learning rate with warmup
3. Gradient clipping
4. LSQ-style learnable scale (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {DEVICE}")

# =============================================================================
# Quantization - Q3.4 (ap_fixed<8,4>)
# =============================================================================
SCALE = 16
QMIN = -128
QMAX = 127
REAL_MIN = -8.0
REAL_MAX = 7.9375

print(f"Quantization: Q3.4 (ap_fixed<8,4>)")
print(f"Range: [{REAL_MIN}, {REAL_MAX}]")


class STEQuantize(torch.autograd.Function):
    """STE with gradient clipping for stability"""
    @staticmethod
    def forward(ctx, x):
        # Save for backward (for gradient masking)
        ctx.save_for_backward(x)
        x_clamped = torch.clamp(x, REAL_MIN, REAL_MAX)
        x_int = torch.round(x_clamped * SCALE)
        x_int = torch.clamp(x_int, QMIN, QMAX)
        return x_int / SCALE

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Gradient masking: zero gradients for values outside range
        # This helps with stability
        grad_input = grad_output.clone()
        grad_input[x < REAL_MIN] = 0
        grad_input[x > REAL_MAX] = 0
        return grad_input


def quantize(x):
    return STEQuantize.apply(x)


def quantize_tensor(x):
    with torch.no_grad():
        x_clamped = torch.clamp(x, REAL_MIN, REAL_MAX)
        x_int = torch.round(x_clamped * SCALE)
        x_int = torch.clamp(x_int, QMIN, QMAX)
        return x_int / SCALE


# =============================================================================
# Soft Quantization (for progressive training)
# =============================================================================

def soft_quantize(x, temperature=1.0):
    """
    Soft quantization that becomes hard as temperature -> 0
    Allows gradual transition from FP32 to quantized
    """
    if temperature >= 10.0:
        return x  # No quantization
    elif temperature <= 0.01:
        return quantize(x)  # Hard quantization
    else:
        # Blend between original and quantized
        x_q = quantize(x)
        alpha = 1.0 / (1.0 + temperature)
        return alpha * x_q + (1 - alpha) * x


# =============================================================================
# Fire Module with Progressive Quantization
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

    def forward(self, x, quant_weights=False, quant_acts=False, temperature=1.0):
        if quant_weights:
            w_sq = quantize(self.squeeze.weight)
            b_sq = quantize(self.squeeze.bias)
            x = F.conv2d(x, w_sq, b_sq)
        else:
            x = self.squeeze(x)

        x = F.relu(x)
        if quant_acts:
            x = soft_quantize(x, temperature)

        if quant_weights:
            w_e1 = quantize(self.expand1x1.weight)
            b_e1 = quantize(self.expand1x1.bias)
            o1 = F.conv2d(x, w_e1, b_e1)

            w_e3 = quantize(self.expand3x3.weight)
            b_e3 = quantize(self.expand3x3.bias)
            o3 = F.conv2d(x, w_e3, b_e3, padding=1)
        else:
            o1 = self.expand1x1(x)
            o3 = self.expand3x3(x)

        if quant_acts:
            o1 = soft_quantize(o1, temperature)
            o3 = soft_quantize(o3, temperature)

        out = torch.cat([o1, o3], dim=1)
        out = F.relu(out)

        return out


# =============================================================================
# SqueezeNet with Progressive Quantization
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

    def forward(self, x, quant_weights=False, quant_acts=False, temperature=1.0):
        # Quantize input
        if quant_acts:
            x = soft_quantize(x, temperature)

        # Conv1
        if quant_weights:
            w1 = quantize(self.conv1.weight)
            b1 = quantize(self.conv1.bias)
            x = F.conv2d(x, w1, b1, stride=2, padding=3)
        else:
            x = self.conv1(x)

        x = F.relu(x)
        if quant_acts:
            x = soft_quantize(x, temperature)

        x = self.pool(x)

        # Fire modules
        x = self.fire2(x, quant_weights, quant_acts, temperature)
        x = self.fire3(x, quant_weights, quant_acts, temperature)
        x = self.fire4(x, quant_weights, quant_acts, temperature)
        x = self.pool(x)

        x = self.fire5(x, quant_weights, quant_acts, temperature)
        x = self.fire6(x, quant_weights, quant_acts, temperature)
        x = self.fire7(x, quant_weights, quant_acts, temperature)
        x = self.fire8(x, quant_weights, quant_acts, temperature)
        x = self.pool(x)

        x = self.fire9(x, quant_weights, quant_acts, temperature)

        # Conv10
        if quant_weights:
            w10 = quantize(self.conv10.weight)
            b10 = quantize(self.conv10.bias)
            x = F.conv2d(x, w10, b10)
        else:
            x = self.conv10(x)

        x = F.relu(x)
        if quant_acts:
            x = soft_quantize(x, temperature)

        x = self.avg(x)
        if quant_acts:
            x = soft_quantize(x, temperature)

        return torch.flatten(x, 1)


# =============================================================================
# Data
# =============================================================================

transform_train = T.Compose([
    T.Resize(224),
    T.RandomHorizontalFlip(),
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
    batch_size=64, shuffle=True, num_workers=2, pin_memory=True  # Smaller batch for stability
)
testloader = DataLoader(
    torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform_test),
    batch_size=128, shuffle=False, num_workers=2, pin_memory=True
)


# =============================================================================
# Test Functions
# =============================================================================

def test(model, loader, quant_weights=False, quant_acts=False):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x, quant_weights=quant_weights, quant_acts=quant_acts, temperature=0.0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


# =============================================================================
# Export
# =============================================================================

def export_weights(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_weights = []
    all_biases = []

    layers = [
        ('conv1', model.conv1),
        ('fire2_squeeze', model.fire2.squeeze),
        ('fire2_expand1x1', model.fire2.expand1x1),
        ('fire2_expand3x3', model.fire2.expand3x3),
        ('fire3_squeeze', model.fire3.squeeze),
        ('fire3_expand1x1', model.fire3.expand1x1),
        ('fire3_expand3x3', model.fire3.expand3x3),
        ('fire4_squeeze', model.fire4.squeeze),
        ('fire4_expand1x1', model.fire4.expand1x1),
        ('fire4_expand3x3', model.fire4.expand3x3),
        ('fire5_squeeze', model.fire5.squeeze),
        ('fire5_expand1x1', model.fire5.expand1x1),
        ('fire5_expand3x3', model.fire5.expand3x3),
        ('fire6_squeeze', model.fire6.squeeze),
        ('fire6_expand1x1', model.fire6.expand1x1),
        ('fire6_expand3x3', model.fire6.expand3x3),
        ('fire7_squeeze', model.fire7.squeeze),
        ('fire7_expand1x1', model.fire7.expand1x1),
        ('fire7_expand3x3', model.fire7.expand3x3),
        ('fire8_squeeze', model.fire8.squeeze),
        ('fire8_expand1x1', model.fire8.expand1x1),
        ('fire8_expand3x3', model.fire8.expand3x3),
        ('fire9_squeeze', model.fire9.squeeze),
        ('fire9_expand1x1', model.fire9.expand1x1),
        ('fire9_expand3x3', model.fire9.expand3x3),
        ('conv10', model.conv10),
    ]

    print("\nExporting:")
    for name, layer in layers:
        w = quantize_tensor(layer.weight.data).cpu().numpy()
        b = quantize_tensor(layer.bias.data).cpu().numpy()
        w_int = np.round(w * SCALE).astype(np.int8)
        b_int = np.round(b * SCALE).astype(np.int8)
        all_weights.extend(w_int.flatten().tolist())
        all_biases.extend(b_int.flatten().tolist())
        print(f"  {name}: {w.shape}")

    np.array(all_weights, dtype=np.int8).tofile(os.path.join(output_dir, 'weights.bin'))
    np.array(all_biases, dtype=np.int8).tofile(os.path.join(output_dir, 'biases.bin'))
    print(f"\nSaved: {output_dir}/weights.bin ({len(all_weights):,} weights)")
    print(f"Saved: {output_dir}/biases.bin ({len(all_biases):,} biases)")


# =============================================================================
# Main - Progressive QAT
# =============================================================================

def main():
    print("\n" + "="*70)
    print("EE 511 - Task 2: Stable Progressive QAT")
    print("="*70)

    os.makedirs("checkpoints", exist_ok=True)

    model = SqueezeNet(num_classes=10).to(DEVICE)

    # Load Task 1 checkpoint
    ckpt_path = "checkpoints/task1_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: {ckpt_path} not found!")
        return

    print(f"\nLoading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])

    # Baseline
    fp32_acc = test(model, testloader, quant_weights=False, quant_acts=False)
    weight_only_acc = test(model, testloader, quant_weights=True, quant_acts=False)
    full_quant_acc = test(model, testloader, quant_weights=True, quant_acts=True)

    print(f"\nBaseline Results:")
    print(f"  FP32:              {fp32_acc:.2f}%")
    print(f"  Weight-only PTQ:   {weight_only_acc:.2f}%")
    print(f"  Full Quant PTQ:    {full_quant_acc:.2f}%")

    # ==========================================================================
    # Phase 1: Weight-only QAT (stabilize weights first)
    # ==========================================================================
    print("\n" + "="*70)
    print("Phase 1: Weight-Only QAT (5 epochs)")
    print("="*70)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, 6):
        model.train()
        running_loss = correct = total = 0

        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x, quant_weights=True, quant_acts=False)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = 100.0 * correct / total
        test_acc = test(model, testloader, quant_weights=True, quant_acts=False)
        print(f"  Epoch {epoch}: Train {train_acc:.2f}%, Test {test_acc:.2f}%")

    # ==========================================================================
    # Phase 2: Progressive Full Quantization
    # ==========================================================================
    print("\n" + "="*70)
    print("Phase 2: Progressive Activation Quantization (15 epochs)")
    print("="*70)

    optimizer = optim.Adam(model.parameters(), lr=5e-5)  # Lower LR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

    history = []
    best_acc = 0

    # Temperature schedule: starts high (soft), decreases to 0 (hard)
    # High temp = more like FP32, Low temp = hard quantization
    temp_schedule = [5.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0]

    print(f"\n{'Epoch':<7} {'Temp':<8} {'Loss':<10} {'Train':<10} {'Test':<10} {'Status'}")
    print("-"*70)

    for epoch in range(1, 16):
        temperature = temp_schedule[epoch - 1]

        model.train()
        running_loss = correct = total = 0

        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x, quant_weights=True, quant_acts=True, temperature=temperature)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Aggressive clipping
            optimizer.step()

            running_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        scheduler.step()

        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * correct / total
        # Test with HARD quantization (temperature=0)
        test_acc = test(model, testloader, quant_weights=True, quant_acts=True)

        history.append({'epoch': epoch, 'train_acc': train_acc, 'test_acc': test_acc, 'temp': temperature})

        status = ""
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_acc': test_acc},
                      "checkpoints/task2_best.pth")
            status = f"★ Best: {best_acc:.2f}%"

        print(f"{epoch:<7} {temperature:<8.2f} {train_loss:<10.4f} {train_acc:<10.2f} {test_acc:<10.2f} {status}")

    # ==========================================================================
    # Phase 3: Fine-tuning with hard quantization
    # ==========================================================================
    print("\n" + "="*70)
    print("Phase 3: Fine-tuning with Hard Quantization (10 epochs)")
    print("="*70)

    # Load best model from Phase 2
    best_ckpt = torch.load("checkpoints/task2_best.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(best_ckpt['model_state_dict'])

    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

    print(f"\n{'Epoch':<7} {'Loss':<10} {'Train':<10} {'Test':<10} {'Status'}")
    print("-"*60)

    for epoch in range(1, 11):
        model.train()
        running_loss = correct = total = 0

        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x, quant_weights=True, quant_acts=True, temperature=0.0)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            running_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * correct / total
        test_acc = test(model, testloader, quant_weights=True, quant_acts=True)

        history.append({'epoch': 15 + epoch, 'train_acc': train_acc, 'test_acc': test_acc, 'temp': 0.0})

        status = ""
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model_state_dict': model.state_dict(), 'test_acc': test_acc},
                      "checkpoints/task2_best.pth")
            status = f"★ Best: {best_acc:.2f}%"

        print(f"{epoch:<7} {train_loss:<10.4f} {train_acc:<10.2f} {test_acc:<10.2f} {status}")

    # Results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"FP32 Baseline:      {fp32_acc:.2f}%")
    print(f"Weight-only PTQ:    {weight_only_acc:.2f}%")
    print(f"Full Quant PTQ:     {full_quant_acc:.2f}%")
    print(f"Full Quant QAT:     {best_acc:.2f}%")
    print(f"Accuracy Drop:      {fp32_acc - best_acc:.2f}%")
    print("="*70)

    # Export
    best_ckpt = torch.load("checkpoints/task2_best.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(best_ckpt['model_state_dict'])
    export_weights(model, "checkpoints/hls_weights")

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    epochs = [h['epoch'] for h in history]
    plt.plot(epochs, [h['train_acc'] for h in history], 'b-', label='Train')
    plt.plot(epochs, [h['test_acc'] for h in history], 'r-', label='Test')
    plt.axhline(y=fp32_acc, color='g', linestyle='--', label=f'FP32 ({fp32_acc:.1f}%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Progressive QAT Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    temps = [h['temp'] for h in history]
    plt.plot(epochs, temps, 'purple', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Temperature')
    plt.title('Quantization Temperature Schedule')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("checkpoints/qat_progressive.png", dpi=150)
    plt.savefig("checkpoints/qat_progressive.pdf")
    plt.show()

    print("\nDone!")


if __name__ == '__main__':
    main()
