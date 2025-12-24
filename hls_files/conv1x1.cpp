#include "conv1x1.h"

// ============================================================================
// 1x1 Convolution Implementation - OPTIMIZED FOR II=1 with Q3.4 format
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
) {
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=1605632 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=1605632 max_write_burst_length=256
    #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem2 depth=131072 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem3 depth=512 max_read_burst_length=256
    #pragma HLS INTERFACE s_axilite port=in_channels
    #pragma HLS INTERFACE s_axilite port=out_channels
    #pragma HLS INTERFACE s_axilite port=height
    #pragma HLS INTERFACE s_axilite port=width
    #pragma HLS INTERFACE s_axilite port=enable
    #pragma HLS INTERFACE s_axilite port=apply_relu
    #pragma HLS INTERFACE s_axilite port=return

    if (!enable) return;

    const int HW = height * width;

    // Weight buffer for one output channel - optimized partitioning
    weight_t w_buf[512];
    #pragma HLS ARRAY_PARTITION variable=w_buf cyclic factor=16
    #pragma HLS BIND_STORAGE variable=w_buf type=ram_2p impl=bram

    CONV1_OC:
    for (int oc = 0; oc < out_channels; oc++) {
        #pragma HLS LOOP_TRIPCOUNT min=256 max=512

        // Load weights for this output channel
        LOAD_W:
        for (int ic = 0; ic < in_channels; ic++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=128 max=512
            if (ic < 512) {
                w_buf[ic] = weights[oc * in_channels + ic];
            }
        }

        bias_t b = bias[oc];

        CONV1_H:
        for (int h = 0; h < height; h++) {
            #pragma HLS LOOP_TRIPCOUNT min=7 max=56

            CONV1_W:
            for (int w = 0; w < width; w++) {
                #pragma HLS LOOP_TRIPCOUNT min=7 max=56

                // For 1x1 convolution, dependency is less severe
                // but we can still optimize with proper pragmas
                acc_t sum = (acc_t)b;

                CONV1_IC:
                for (int ic = 0; ic < in_channels; ic++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=128 max=512
                    #pragma HLS DEPENDENCE variable=sum inter false

                    int in_idx = ic * HW + h * width + w;
                    if (ic < 512) {
                        acc_t mult_result = (acc_t)input[in_idx] * (acc_t)w_buf[ic];
                        sum += mult_result;
                    }
                }

                // Apply ReLU if requested
                data_t out_val = (data_t)sum;
                if (apply_relu && out_val < 0) out_val = 0;

                int out_idx = oc * HW + h * width + w;
                output[out_idx] = out_val;
            }
        }
    }
}

// ============================================================================
// 1x1 Convolution Without ReLU (for final classifier layer)
// ============================================================================

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
) {
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=1605632 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=1000 max_write_burst_length=256
    #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem2 depth=512000 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem3 depth=1000 max_read_burst_length=256
    #pragma HLS INTERFACE s_axilite port=in_channels
    #pragma HLS INTERFACE s_axilite port=out_channels
    #pragma HLS INTERFACE s_axilite port=height
    #pragma HLS INTERFACE s_axilite port=width
    #pragma HLS INTERFACE s_axilite port=enable
    #pragma HLS INTERFACE s_axilite port=return

    if (!enable) return;

    const int HW = height * width;

    // Weight buffer
    weight_t w_buf[512];
    #pragma HLS ARRAY_PARTITION variable=w_buf cyclic factor=16
    #pragma HLS BIND_STORAGE variable=w_buf type=ram_2p impl=bram

    CONV1_NR_OC:
    for (int oc = 0; oc < out_channels; oc++) {
        #pragma HLS LOOP_TRIPCOUNT min=1000 max=1000

        // Load weights for this output channel
        LOAD_NR_W:
        for (int ic = 0; ic < in_channels; ic++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=512 max=512
            if (ic < 512) {
                w_buf[ic] = weights[oc * in_channels + ic];
            }
        }

        bias_t b = bias[oc];

        CONV1_NR_H:
        for (int h = 0; h < height; h++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1

            CONV1_NR_W:
            for (int w = 0; w < width; w++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=1

                acc_t sum = (acc_t)b;

                CONV1_NR_IC:
                for (int ic = 0; ic < in_channels; ic++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=512 max=512
                    #pragma HLS DEPENDENCE variable=sum inter false

                    int in_idx = ic * HW + h * width + w;
                    if (ic < 512) {
                        acc_t mult_result = (acc_t)input[in_idx] * (acc_t)w_buf[ic];
                        sum += mult_result;
                    }
                }

                // No ReLU for classifier output
                int out_idx = oc * HW + h * width + w;
                output[out_idx] = (data_t)sum;
            }
        }
    }
}
