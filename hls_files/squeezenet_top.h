#ifndef SQUEEZENET_TOP_H
#define SQUEEZENET_TOP_H

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include "squeezenet_types.h"  // Contains LayerConfig, FireConfig, and buffer sizes

// ============================================================================
// Buffer Size Definitions
// ============================================================================
// NOTE: WORK_BUF_SIZE and DDR_BUFFER_SIZE are now defined in squeezenet_types.h
// Do NOT redefine them here to avoid conflicts!

// Maximum feature map size: 512 channels × 56 × 56 = 1,605,632 elements
#define MAX_FM_SIZE     (512 * 56 * 56)  // 1,605,632 elements

// Line buffer size for streaming optimizations
#define LINE_BUF_SIZE   (512 * 7)        // 3,584 elements for line buffering

// Weight and bias buffer sizes
#define MAX_WEIGHTS     740000
#define MAX_BIASES      3000

// ============================================================================
// Top-Level Function Declaration
// ============================================================================
void squeezenet_top(
    // Input/Output in DDR
    data_t *input,          // 3×224×224 input image in DDR
    data_t *output,         // 10 class scores output in DDR

    // Weights and biases in DDR
    weight_t *weights,      // All network weights in DDR
    bias_t *biases,         // All network biases in DDR

    // Large working buffers in DDR (feature maps)
    data_t *ddr_buffer_a,   // Primary feature map buffer in DDR
    data_t *ddr_buffer_b,   // Secondary feature map buffer in DDR

    // Control signals
    bool start,
    bool *done
);

#endif // SQUEEZENET_TOP_H
