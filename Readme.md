# SqueezeNet FPGA Accelerator

FPGA implementation of SqueezeNet for CIFAR-10 image classification on Xilinx Kria KV260.

## Overview

- **Model**: SqueezeNet v1.0 with Q3.4 fixed-point quantization
- **Dataset**: CIFAR-10 (10 classes, 224×224 input)
- **Platform**: Xilinx Kria KV260, PYNQ framework
- **Accuracy**: ~90% on test images

## Project Structure

```
├── Training_code/
│   ├── task1.py          # FP32 baseline training
│   ├── task2.py          # Quantization-aware training (QAT)
│   └── test_quant.py     # HLS-compatible inference test
├── hls_files/
│   ├── squeezenet_top.cpp/h    # Top-level accelerator
│   ├── controller.cpp/h        # FSM controller
│   ├── fire_module.cpp/h       # Fire module (squeeze + expand)
│   ├── conv1x1.cpp/h           # 1×1 convolution
│   ├── conv3x3.cpp/h           # 3×3 and 7×7 convolution
│   ├── maxpool.cpp/h           # MaxPool and Global AvgPool
│   ├── squeezenet_types.h      # Q3.4 type definitions
│   └── tb_squeezenet.cpp       # C++ testbench
└── host_code.ipynb             # PYNQ host application
```

## Quick Start

### 1. Training
```bash
cd Training_code
python task1.py   # FP32 training (100 epochs)
python task2.py   # QAT with progressive quantization
```

### 2. HLS Synthesis
Run in Vitis HLS with `squeezenet_top` as top function, then export IP.

### 3. Vivado Implementation
1. Create new Vivado project targeting Kria KV260
2. Create block design with Zynq UltraScale+ MPSoC
3. Add exported HLS IP (`squeezenet_top`)
4. Run Connection Automation for AXI interfaces
5. Generate HDL wrapper
6. Run Synthesis → Implementation → Generate Bitstream
7. Export hardware (`.hwh`) and bitstream (`.bit`)

### 4. FPGA Deployment
Upload to Kria KV260 with PYNQ:
- `design_1.bit` - Bitstream
- `design_1.hwh` - Hardware
- `weights.bin`, `biases.bin` - Quantized weights
- `host_code.ipynb` - Inference notebook

## Key Features

- **Controller FSM**: Sequential layer execution with enable signals
- **DDR double-buffering**: Ping-pong buffers for feature maps
- **Q3.4 quantization**: 8-bit fixed-point (`ap_fixed<8,4>`)
- **Optimized HLS**: Tree reduction, burst transfers, II=1 pipelines

## Requirements

- Vitis HLS 2022.1+
- PYNQ 2.7+
- PyTorch 2.0+
