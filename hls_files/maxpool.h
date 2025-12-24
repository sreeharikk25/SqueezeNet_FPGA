#ifndef MAXPOOL_H
#define MAXPOOL_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include "squeezenet_types.h"  // Use project's type definitions (Q3.4 format)

// ============================================================================
// Function Declarations
// ============================================================================

void maxpool(
    data_t *input,
    data_t *output,
    int channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    bool ceil_mode,
    bool enable
);

void global_avgpool(
    data_t *input,
    data_t *output,
    int in_channels,
    int height,
    int width,
    bool enable
);

void spatial_dropout(
    data_t *input,
    data_t *output,
    int channels,
    int height,
    int width,
    float drop_rate,
    bool enable
);

#endif // MAXPOOL_H
