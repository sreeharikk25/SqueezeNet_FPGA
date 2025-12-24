#include "fire_module.h"

// ============================================================================
// Fire Module Implementation - FULLY OPTIMIZED
// - Uses external DDR squeeze buffer (fixes URAM accuracy issues)
// - Burst-buffered ReLU for II=1 (fixes pipeline violation)
// ============================================================================

// Maximum dimensions for local buffers
#define MAX_SQ_CH 64
#define MAX_HW (56 * 56)

// ============================================================================
// Internal Squeeze Layer - OPTIMIZED
// Writes to external squeeze_buffer (DDR) instead of internal URAM
// ============================================================================

static void squeeze_layer(
    data_t *input,
    data_t *output,        // External DDR squeeze buffer
    weight_t *weights,
    bias_t *bias,
    int in_channels,
    int out_channels,
    int height,
    int width
) {
    const int HW = height * width;

    // Weight buffer for one output channel with better partitioning
    weight_t w_buf[512];
    #pragma HLS ARRAY_PARTITION variable=w_buf cyclic factor=16
    #pragma HLS BIND_STORAGE variable=w_buf type=ram_2p impl=bram

    SQUEEZE_OC:
    for (int oc = 0; oc < out_channels; oc++) {
        #pragma HLS LOOP_TRIPCOUNT min=16 max=64

        // Load weights for this output channel
        LOAD_SQ_W:
        for (int ic = 0; ic < in_channels; ic++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=96 max=512
            if (ic < 512) {
                w_buf[ic] = weights[oc * in_channels + ic];
            }
        }

        bias_t b = bias[oc];

        SQUEEZE_H:
        for (int h = 0; h < height; h++) {
            #pragma HLS LOOP_TRIPCOUNT min=7 max=56

            SQUEEZE_W:
            for (int w = 0; w < width; w++) {
                #pragma HLS LOOP_TRIPCOUNT min=7 max=56

                acc_t sum = (acc_t)b;

                SQUEEZE_IC:
                for (int ic = 0; ic < in_channels; ic++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=96 max=512
                    #pragma HLS DEPENDENCE variable=sum inter false

                    int in_idx = ic * HW + h * width + w;
                    if (ic < 512) {
                        acc_t mult_result = (acc_t)input[in_idx] * (acc_t)w_buf[ic];
                        sum += mult_result;
                    }
                }

                // ReLU
                data_t out_val = (data_t)sum;
                if (out_val < 0) out_val = 0;

                int out_idx = oc * HW + h * width + w;
                output[out_idx] = out_val;
            }
        }
    }
}

// ============================================================================
// Internal Expand 1x1 Layer - OPTIMIZED
// Reads from external squeeze_buffer (DDR)
// ============================================================================

static void expand1x1_layer(
    data_t *input,         // External DDR squeeze buffer
    data_t *output,
    weight_t *weights,
    bias_t *bias,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int output_channel_offset
) {
    const int HW = height * width;

    // Weight buffer for one output channel
    weight_t w_buf[64];
    #pragma HLS ARRAY_PARTITION variable=w_buf cyclic factor=8
    #pragma HLS BIND_STORAGE variable=w_buf type=ram_2p impl=bram

    EXPAND1_OC:
    for (int oc = 0; oc < out_channels; oc++) {
        #pragma HLS LOOP_TRIPCOUNT min=64 max=256

        // Load weights for this output channel
        LOAD_E1_W:
        for (int ic = 0; ic < in_channels; ic++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=16 max=64
            if (ic < 64) {
                w_buf[ic] = weights[oc * in_channels + ic];
            }
        }

        bias_t b = bias[oc];

        EXPAND1_H:
        for (int h = 0; h < height; h++) {
            #pragma HLS LOOP_TRIPCOUNT min=7 max=56

            EXPAND1_W:
            for (int w = 0; w < width; w++) {
                #pragma HLS LOOP_TRIPCOUNT min=7 max=56

                acc_t sum = (acc_t)b;

                EXPAND1_IC:
                for (int ic = 0; ic < in_channels; ic++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=16 max=64
                    #pragma HLS DEPENDENCE variable=sum inter false

                    int in_idx = ic * HW + h * width + w;
                    if (ic < 64) {
                        acc_t mult_result = (acc_t)input[in_idx] * (acc_t)w_buf[ic];
                        sum += mult_result;
                    }
                }

                int out_idx = (output_channel_offset + oc) * HW + h * width + w;
                output[out_idx] = (data_t)sum;
            }
        }
    }
}

// ============================================================================
// Internal Expand 3x3 Layer - OPTIMIZED WITH TREE REDUCTION
// Reads from external squeeze_buffer (DDR)
// ============================================================================

static void expand3x3_layer(
    data_t *input,         // External DDR squeeze buffer
    data_t *output,
    weight_t *weights,
    bias_t *bias,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int output_channel_offset
) {
    const int K = 3;
    const int PAD = 1;
    const int HW = height * width;

    // Weight buffer for one output channel - optimized partitioning
    weight_t w_buf[64][3][3];
    #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=3
    #pragma HLS ARRAY_PARTITION variable=w_buf cyclic factor=8 dim=1

    // Input window buffer - optimized partitioning
    data_t in_window[64][3][3];
    #pragma HLS ARRAY_PARTITION variable=in_window complete dim=2
    #pragma HLS ARRAY_PARTITION variable=in_window complete dim=3
    #pragma HLS ARRAY_PARTITION variable=in_window cyclic factor=8 dim=1

    EXPAND3_OC:
    for (int oc = 0; oc < out_channels; oc++) {
        #pragma HLS LOOP_TRIPCOUNT min=64 max=256

        // Load weights for this output channel
        E3_LOAD_IC:
        for (int ic = 0; ic < in_channels; ic++) {
            #pragma HLS LOOP_TRIPCOUNT min=16 max=64
            E3_LOAD_KH:
            for (int kh = 0; kh < K; kh++) {
                E3_LOAD_KW:
                for (int kw = 0; kw < K; kw++) {
                    #pragma HLS PIPELINE II=1
                    int w_idx = oc * in_channels * K * K + ic * K * K + kh * K + kw;
                    if (ic < 64) {
                        w_buf[ic][kh][kw] = weights[w_idx];
                    }
                }
            }
        }

        bias_t b = bias[oc];

        EXPAND3_OH:
        for (int oh = 0; oh < height; oh++) {
            #pragma HLS LOOP_TRIPCOUNT min=7 max=56

            EXPAND3_OW:
            for (int ow = 0; ow < width; ow++) {
                #pragma HLS LOOP_TRIPCOUNT min=7 max=56

                // Load input window into local buffer first
                E3_LOAD_WIN_IC:
                for (int ic = 0; ic < in_channels; ic++) {
                    #pragma HLS LOOP_TRIPCOUNT min=16 max=64
                    E3_LOAD_WIN_KH:
                    for (int kh = 0; kh < K; kh++) {
                        E3_LOAD_WIN_KW:
                        for (int kw = 0; kw < K; kw++) {
                            #pragma HLS PIPELINE II=1

                            int ih = oh + kh - PAD;
                            int iw = ow + kw - PAD;

                            data_t val = 0;
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                val = input[ic * HW + ih * width + iw];
                            }
                            if (ic < 64) {
                                in_window[ic][kh][kw] = val;
                            }
                        }
                    }
                }

                // ================================================================
                // KEY FIX: Tree reduction to achieve II=1
                // ================================================================

                // Partial sums for each of the 9 kernel positions
                acc_t partial_sums[9];
                #pragma HLS ARRAY_PARTITION variable=partial_sums complete

                // Initialize partial sums
                INIT_PARTIAL:
                for (int i = 0; i < 9; i++) {
                    #pragma HLS UNROLL
                    partial_sums[i] = 0;
                }

                // Accumulate each input channel's contribution to all 9 positions
                EXPAND3_IC:
                for (int ic = 0; ic < in_channels; ic++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=16 max=64
                    #pragma HLS DEPENDENCE variable=partial_sums inter false
                    #pragma HLS DEPENDENCE variable=in_window inter false
                    #pragma HLS DEPENDENCE variable=w_buf inter false

                    if (ic < 64) {
                        partial_sums[0] += (acc_t)in_window[ic][0][0] * (acc_t)w_buf[ic][0][0];
                        partial_sums[1] += (acc_t)in_window[ic][0][1] * (acc_t)w_buf[ic][0][1];
                        partial_sums[2] += (acc_t)in_window[ic][0][2] * (acc_t)w_buf[ic][0][2];
                        partial_sums[3] += (acc_t)in_window[ic][1][0] * (acc_t)w_buf[ic][1][0];
                        partial_sums[4] += (acc_t)in_window[ic][1][1] * (acc_t)w_buf[ic][1][1];
                        partial_sums[5] += (acc_t)in_window[ic][1][2] * (acc_t)w_buf[ic][1][2];
                        partial_sums[6] += (acc_t)in_window[ic][2][0] * (acc_t)w_buf[ic][2][0];
                        partial_sums[7] += (acc_t)in_window[ic][2][1] * (acc_t)w_buf[ic][2][1];
                        partial_sums[8] += (acc_t)in_window[ic][2][2] * (acc_t)w_buf[ic][2][2];
                    }
                }

                // Final tree reduction of partial sums
                acc_t sum = (acc_t)b;
                #pragma HLS BIND_OP variable=sum op=add impl=fabric

                acc_t level1_0 = partial_sums[0] + partial_sums[1];
                acc_t level1_1 = partial_sums[2] + partial_sums[3];
                acc_t level1_2 = partial_sums[4] + partial_sums[5];
                acc_t level1_3 = partial_sums[6] + partial_sums[7];

                acc_t level2_0 = level1_0 + level1_1;
                acc_t level2_1 = level1_2 + level1_3;

                sum += level2_0 + level2_1 + partial_sums[8];

                int out_idx = (output_channel_offset + oc) * HW + oh * width + ow;
                output[out_idx] = (data_t)sum;
            }
        }
    }
}

// ============================================================================
// Apply ReLU In-Place - FIXED with Burst Buffering for II=1
// Uses local BRAM buffer to avoid DDR read-modify-write latency
// ============================================================================

static void apply_relu_inplace(
    data_t *data,
    int channels,
    int height,
    int width
) {
    int total = channels * height * width;

    // Local buffer for burst transfers - sized for efficient DDR access
    const int BURST_LEN = 256;
    data_t local_buf[BURST_LEN];
    #pragma HLS ARRAY_PARTITION variable=local_buf cyclic factor=16

    // Process data in chunks
    int num_full_bursts = total / BURST_LEN;
    int remainder = total % BURST_LEN;

    // Process full bursts
    RELU_BURST:
    for (int b = 0; b < num_full_bursts; b++) {
        #pragma HLS LOOP_TRIPCOUNT min=39 max=392

        int base_addr = b * BURST_LEN;

        // Burst read from DDR to local buffer
        RELU_READ:
        for (int i = 0; i < BURST_LEN; i++) {
            #pragma HLS PIPELINE II=1
            local_buf[i] = data[base_addr + i];
        }

        // Apply ReLU on local buffer (II=1 achievable on BRAM)
        RELU_COMPUTE:
        for (int i = 0; i < BURST_LEN; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=16
            if (local_buf[i] < data_t(0)) {
                local_buf[i] = data_t(0);
            }
        }

        // Burst write from local buffer to DDR
        RELU_WRITE:
        for (int i = 0; i < BURST_LEN; i++) {
            #pragma HLS PIPELINE II=1
            data[base_addr + i] = local_buf[i];
        }
    }

    // Process remainder (if any)
    if (remainder > 0) {
        int base_addr = num_full_bursts * BURST_LEN;

        // Read remainder
        RELU_READ_REM:
        for (int i = 0; i < remainder; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=255
            local_buf[i] = data[base_addr + i];
        }

        // Apply ReLU on remainder
        RELU_COMPUTE_REM:
        for (int i = 0; i < remainder; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=255
            if (local_buf[i] < data_t(0)) {
                local_buf[i] = data_t(0);
            }
        }

        // Write remainder
        RELU_WRITE_REM:
        for (int i = 0; i < remainder; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=255
            data[base_addr + i] = local_buf[i];
        }
    }
}

// ============================================================================
// Main Fire Module Function - OPTIMIZED with External Squeeze Buffer
// ============================================================================

void fire_module(
    data_t *input,
    data_t *output,
    data_t *squeeze_buffer,    // External DDR buffer for squeeze output
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
) {
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=802816 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=802816 max_write_burst_length=256
    #pragma HLS INTERFACE m_axi port=squeeze_buffer offset=slave bundle=gmem4 depth=200704 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS INTERFACE m_axi port=squeeze_weights offset=slave bundle=gmem2 depth=32768 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=squeeze_bias offset=slave bundle=gmem3 depth=64 max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=expand1x1_weights offset=slave bundle=gmem2 depth=16384 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=expand1x1_bias offset=slave bundle=gmem3 depth=256 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=expand3x3_weights offset=slave bundle=gmem2 depth=147456 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=expand3x3_bias offset=slave bundle=gmem3 depth=256 max_read_burst_length=256
    #pragma HLS INTERFACE s_axilite port=in_channels
    #pragma HLS INTERFACE s_axilite port=squeeze_channels
    #pragma HLS INTERFACE s_axilite port=expand1x1_channels
    #pragma HLS INTERFACE s_axilite port=expand3x3_channels
    #pragma HLS INTERFACE s_axilite port=height
    #pragma HLS INTERFACE s_axilite port=width
    #pragma HLS INTERFACE s_axilite port=enable
    #pragma HLS INTERFACE s_axilite port=return

    if (!enable) return;

    // Step 1: Squeeze layer (1x1 conv + ReLU) -> writes to external squeeze_buffer
    squeeze_layer(
        input, squeeze_buffer,
        squeeze_weights, squeeze_bias,
        in_channels, squeeze_channels,
        height, width
    );

    // Step 2: Expand 1x1 layer (reads from squeeze_buffer, writes to output[0:e1_ch])
    expand1x1_layer(
        squeeze_buffer, output,
        expand1x1_weights, expand1x1_bias,
        squeeze_channels, expand1x1_channels,
        height, width,
        0
    );

    // Step 3: Expand 3x3 layer (reads from squeeze_buffer, writes to output[e1_ch:e1_ch+e3_ch])
    expand3x3_layer(
        squeeze_buffer, output,
        expand3x3_weights, expand3x3_bias,
        squeeze_channels, expand3x3_channels,
        height, width,
        expand1x1_channels
    );

    // Step 4: Apply ReLU to concatenated output (burst-buffered for II=1)
    int total_out_channels = expand1x1_channels + expand3x3_channels;
    apply_relu_inplace(output, total_out_channels, height, width);
}
