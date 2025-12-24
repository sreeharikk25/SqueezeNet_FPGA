#ifndef SQUEEZENET_TYPES_H
#define SQUEEZENET_TYPES_H

#include <ap_fixed.h>

// ============================================================================
// Fixed-Point Type Definitions for Q3.4 Format
// ============================================================================
//
// IMPORTANT: Using AP_RND_CONV (Banker's Rounding) to match PyTorch torch.round()
// Using AP_SAT (Saturation) to match torch.clamp()
//
// Template: ap_fixed<W, I, Q, O>
//   W = total width in bits
//   I = integer bits (including sign bit)
//   Q = quantization mode (rounding behavior)
//   O = overflow mode (saturation behavior)
//
// Q3.4 format: 1 sign bit + 3 integer bits + 4 fractional bits = 8 bits
// Range: [-8.0, 7.9375], Resolution: 0.0625 (1/16)

typedef ap_fixed<8, 4, AP_RND_CONV, AP_SAT> data_t;      // Activations
typedef ap_fixed<8, 4, AP_RND_CONV, AP_SAT> weight_t;    // Weights
typedef ap_fixed<8, 4, AP_RND_CONV, AP_SAT> bias_t;      // Biases

// Accumulator: wider to prevent overflow during MAC operations
// 24 bits total, 12 integer bits -> supports accumulation of many products
// Rounding mode matters when this gets cast to data_t
typedef ap_fixed<24, 12, AP_RND_CONV, AP_SAT> acc_t;

// ============================================================================
// Utility macros
// ============================================================================
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// ============================================================================
// Configuration Structures
// ============================================================================

typedef struct {
    int in_channels;
    int out_channels;
    int in_height;
    int in_width;
    int kernel_size;
    int stride;
    int padding;
} LayerConfig;

typedef struct {
    int in_channels;
    int squeeze_channels;
    int expand1x1_channels;
    int expand3x3_channels;
    int height;
    int width;
} FireConfig;

// ============================================================================
// Network Dimension Constants
// ============================================================================

#define INPUT_HEIGHT    224
#define INPUT_WIDTH     224
#define INPUT_CHANNELS  3
#define NUM_CLASSES     10

#define WORK_BUF_SIZE   100352
#define DDR_BUFFER_SIZE 1605632

#define MAX_CHANNELS    512
#define MAX_HEIGHT      112
#define MAX_WIDTH       112

#define MAX_SQUEEZE_CH  64
#define MAX_EXPAND_CH   256

#endif // SQUEEZENET_TYPES_H
