#ifndef FIRE_MODULE_H
#define FIRE_MODULE_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "squeezenet_types.h"  // Use project's type definitions (Q3.4 format)

// ============================================================================
// Fire Module Function Declaration
// Updated with external squeeze_buffer parameter for DDR-based intermediate storage
// ============================================================================

void fire_module(
    data_t *input,
    data_t *output,
    data_t *squeeze_buffer,    // External DDR buffer for squeeze output (NEW)
    weight_t *squeeze_weights,
    bias_t *squeeze_bias,
    weight_t *expand1x1_weights,
    bias_t *expand1x1_bias,
    weight_t *expand3x3_weights,
    bias_t *expand3x3_bias,
    int in_channels,
    int squeeze_channels,
    int expand1x1_channels,
    int expand3x3_channels,
    int height,
    int width,
    bool enable
);

#endif // FIRE_MODULE_H
