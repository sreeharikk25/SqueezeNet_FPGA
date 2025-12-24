#ifndef CONV1X1_H
#define CONV1X1_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "squeezenet_types.h"  // Use project's type definitions (Q3.4 format)

// ============================================================================
// Function Declarations
// ============================================================================

void conv1x1(
    data_t *input,
    data_t *output,
    weight_t *weights,
    bias_t *bias,
    int in_channels,
    int out_channels,
    int height,
    int width,
    bool enable,
    bool apply_relu
);

void conv1x1_no_relu(
    data_t *input,
    data_t *output,
    weight_t *weights,
    bias_t *bias,
    int in_channels,
    int out_channels,
    int height,
    int width,
    bool enable
);

#endif // CONV1X1_H
