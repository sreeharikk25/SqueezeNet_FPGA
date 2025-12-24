#ifndef CONV3X3_H
#define CONV3X3_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "squeezenet_types.h"  // Use project's type definitions (Q3.4 format)

// ============================================================================
// Function Declarations
// ============================================================================

void conv3x3(
    data_t *input,
    data_t *output,
    weight_t *weights,
    bias_t *bias,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int stride,
    int padding,
    bool enable,
    bool apply_relu
);

void conv7x7(
    data_t *input,
    data_t *output,
    weight_t *weights,
    bias_t *bias,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int stride,
    int padding,
    bool enable,
    bool apply_relu
);

#endif // CONV3X3_H
