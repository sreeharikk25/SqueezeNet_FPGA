#include "conv3x3.h"

// ============================================================================
// 3x3 Convolution Implementation - OPTIMIZED FOR II=1 with Q3.4 format
// Uses tree reduction to eliminate loop-carried dependencies
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
) {
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=1605632 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=1605632 max_write_burst_length=256
    #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem2 depth=147456 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem3 depth=96 max_read_burst_length=96
    #pragma HLS INTERFACE s_axilite port=in_channels
    #pragma HLS INTERFACE s_axilite port=out_channels
    #pragma HLS INTERFACE s_axilite port=height
    #pragma HLS INTERFACE s_axilite port=width
    #pragma HLS INTERFACE s_axilite port=stride
    #pragma HLS INTERFACE s_axilite port=padding
    #pragma HLS INTERFACE s_axilite port=enable
    #pragma HLS INTERFACE s_axilite port=apply_relu
    #pragma HLS INTERFACE s_axilite port=return

    if (!enable) return;

    const int K = 3;
    int out_height = (height + 2 * padding - K) / stride + 1;
    int out_width = (width + 2 * padding - K) / stride + 1;

    // Weight buffer for one output channel - optimized partitioning
    weight_t w_buf[96][3][3];
    #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=3
    #pragma HLS ARRAY_PARTITION variable=w_buf cyclic factor=8 dim=1

    // Input window buffer - optimized partitioning
    data_t in_window[96][3][3];
    #pragma HLS ARRAY_PARTITION variable=in_window complete dim=2
    #pragma HLS ARRAY_PARTITION variable=in_window complete dim=3
    #pragma HLS ARRAY_PARTITION variable=in_window cyclic factor=8 dim=1

    CONV3_OC:
    for (int oc = 0; oc < out_channels; oc++) {
        #pragma HLS LOOP_TRIPCOUNT min=64 max=96

        // Load weights for this output channel
        CONV3_LOAD_IC:
        for (int ic = 0; ic < in_channels; ic++) {
            #pragma HLS LOOP_TRIPCOUNT min=3 max=96
            CONV3_LOAD_KH:
            for (int kh = 0; kh < K; kh++) {
                CONV3_LOAD_KW:
                for (int kw = 0; kw < K; kw++) {
                    #pragma HLS PIPELINE II=1
                    int w_idx = oc * in_channels * K * K + ic * K * K + kh * K + kw;
                    if (ic < 96) {
                        w_buf[ic][kh][kw] = weights[w_idx];
                    }
                }
            }
        }

        bias_t b = bias[oc];

        CONV3_OH:
        for (int oh = 0; oh < out_height; oh++) {
            #pragma HLS LOOP_TRIPCOUNT min=55 max=111

            CONV3_OW:
            for (int ow = 0; ow < out_width; ow++) {
                #pragma HLS LOOP_TRIPCOUNT min=55 max=111

                // Load input window
                CONV3_LOAD_WIN_IC:
                for (int ic = 0; ic < in_channels; ic++) {
                    #pragma HLS LOOP_TRIPCOUNT min=3 max=96
                    CONV3_LOAD_WIN_KH:
                    for (int kh = 0; kh < K; kh++) {
                        CONV3_LOAD_WIN_KW:
                        for (int kw = 0; kw < K; kw++) {
                            #pragma HLS PIPELINE II=1

                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;

                            data_t val = 0;
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                val = input[ic * height * width + ih * width + iw];
                            }
                            if (ic < 96) {
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
                CONV3_INIT_PARTIAL:
                for (int i = 0; i < 9; i++) {
                    #pragma HLS UNROLL
                    partial_sums[i] = 0;
                }

                // Accumulate with tree reduction
                CONV3_IC:
                for (int ic = 0; ic < in_channels; ic++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=3 max=96
                    #pragma HLS DEPENDENCE variable=partial_sums inter false
                    #pragma HLS DEPENDENCE variable=in_window inter false
                    #pragma HLS DEPENDENCE variable=w_buf inter false

                    if (ic < 96) {
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

                // Final tree reduction
                acc_t sum = (acc_t)b;
                #pragma HLS BIND_OP variable=sum op=add impl=fabric

                acc_t level1_0 = partial_sums[0] + partial_sums[1];
                acc_t level1_1 = partial_sums[2] + partial_sums[3];
                acc_t level1_2 = partial_sums[4] + partial_sums[5];
                acc_t level1_3 = partial_sums[6] + partial_sums[7];

                acc_t level2_0 = level1_0 + level1_1;
                acc_t level2_1 = level1_2 + level1_3;

                sum += level2_0 + level2_1 + partial_sums[8];

                // Apply ReLU if requested
                data_t out_val = (data_t)sum;
                if (apply_relu && out_val < 0) out_val = 0;

                int out_idx = oc * out_height * out_width + oh * out_width + ow;
                output[out_idx] = out_val;
            }
        }
    }
}

// ============================================================================
// 7x7 Convolution Implementation - OPTIMIZED FOR II=1
// ============================================================================

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
) {
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=150528 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=602112 max_write_burst_length=256
    #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem2 depth=14112 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem3 depth=96 max_read_burst_length=96
    #pragma HLS INTERFACE s_axilite port=in_channels
    #pragma HLS INTERFACE s_axilite port=out_channels
    #pragma HLS INTERFACE s_axilite port=height
    #pragma HLS INTERFACE s_axilite port=width
    #pragma HLS INTERFACE s_axilite port=stride
    #pragma HLS INTERFACE s_axilite port=padding
    #pragma HLS INTERFACE s_axilite port=enable
    #pragma HLS INTERFACE s_axilite port=apply_relu
    #pragma HLS INTERFACE s_axilite port=return

    if (!enable) return;

    const int K = 7;
    int out_height = (height + 2 * padding - K) / stride + 1;
    int out_width = (width + 2 * padding - K) / stride + 1;

    // Weight buffer - optimized partitioning
    weight_t w_buf[3][7][7];
    #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=3
    #pragma HLS ARRAY_PARTITION variable=w_buf complete dim=1

    // Input window buffer - optimized partitioning
    data_t in_window[3][7][7];
    #pragma HLS ARRAY_PARTITION variable=in_window complete dim=2
    #pragma HLS ARRAY_PARTITION variable=in_window complete dim=3
    #pragma HLS ARRAY_PARTITION variable=in_window complete dim=1

    CONV7_OC:
    for (int oc = 0; oc < out_channels; oc++) {
        #pragma HLS LOOP_TRIPCOUNT min=64 max=96

        // Load weights for this output channel
        CONV7_LOAD_IC:
        for (int ic = 0; ic < in_channels; ic++) {
            #pragma HLS LOOP_TRIPCOUNT min=3 max=3
            CONV7_LOAD_KH:
            for (int kh = 0; kh < K; kh++) {
                CONV7_LOAD_KW:
                for (int kw = 0; kw < K; kw++) {
                    #pragma HLS PIPELINE II=1
                    int w_idx = oc * in_channels * K * K + ic * K * K + kh * K + kw;
                    if (ic < 3) {
                        w_buf[ic][kh][kw] = weights[w_idx];
                    }
                }
            }
        }

        bias_t b = bias[oc];

        CONV7_OH:
        for (int oh = 0; oh < out_height; oh++) {
            #pragma HLS LOOP_TRIPCOUNT min=111 max=111

            CONV7_OW:
            for (int ow = 0; ow < out_width; ow++) {
                #pragma HLS LOOP_TRIPCOUNT min=111 max=111

                // Load input window
                CONV7_LOAD_WIN_IC:
                for (int ic = 0; ic < in_channels; ic++) {
                    #pragma HLS LOOP_TRIPCOUNT min=3 max=3
                    CONV7_LOAD_WIN_KH:
                    for (int kh = 0; kh < K; kh++) {
                        CONV7_LOAD_WIN_KW:
                        for (int kw = 0; kw < K; kw++) {
                            #pragma HLS PIPELINE II=1

                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;

                            data_t val = 0;
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                val = input[ic * height * width + ih * width + iw];
                            }
                            if (ic < 3) {
                                in_window[ic][kh][kw] = val;
                            }
                        }
                    }
                }

                // ================================================================
                // Tree reduction for 7x7 = 49 kernel elements
                // ================================================================

                acc_t partial_sums[49];
                #pragma HLS ARRAY_PARTITION variable=partial_sums complete

                CONV7_INIT_PARTIAL:
                for (int i = 0; i < 49; i++) {
                    #pragma HLS UNROLL
                    partial_sums[i] = 0;
                }

                CONV7_IC:
                for (int ic = 0; ic < in_channels; ic++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=3 max=3
                    #pragma HLS DEPENDENCE variable=partial_sums inter false
                    #pragma HLS DEPENDENCE variable=in_window inter false
                    #pragma HLS DEPENDENCE variable=w_buf inter false

                    if (ic < 3) {
                        int idx = 0;
                        for (int kh = 0; kh < 7; kh++) {
                            for (int kw = 0; kw < 7; kw++) {
                                partial_sums[idx++] += (acc_t)in_window[ic][kh][kw] * (acc_t)w_buf[ic][kh][kw];
                            }
                        }
                    }
                }

                // Tree reduction for 49 elements
                acc_t sum = (acc_t)b;
                #pragma HLS BIND_OP variable=sum op=add impl=fabric

                // Level 1: 49 -> 25
                acc_t l1[25];
                #pragma HLS ARRAY_PARTITION variable=l1 complete
                for (int i = 0; i < 24; i++) {
                    #pragma HLS UNROLL
                    l1[i] = partial_sums[2*i] + partial_sums[2*i+1];
                }
                l1[24] = partial_sums[48];

                // Level 2: 25 -> 13
                acc_t l2[13];
                #pragma HLS ARRAY_PARTITION variable=l2 complete
                for (int i = 0; i < 12; i++) {
                    #pragma HLS UNROLL
                    l2[i] = l1[2*i] + l1[2*i+1];
                }
                l2[12] = l1[24];

                // Level 3: 13 -> 7
                acc_t l3[7];
                #pragma HLS ARRAY_PARTITION variable=l3 complete
                for (int i = 0; i < 6; i++) {
                    #pragma HLS UNROLL
                    l3[i] = l2[2*i] + l2[2*i+1];
                }
                l3[6] = l2[12];

                // Level 4: 7 -> 4
                acc_t l4[4];
                #pragma HLS ARRAY_PARTITION variable=l4 complete
                for (int i = 0; i < 3; i++) {
                    #pragma HLS UNROLL
                    l4[i] = l3[2*i] + l3[2*i+1];
                }
                l4[3] = l3[6];

                // Final sum
                sum += (l4[0] + l4[1]) + (l4[2] + l4[3]);

                // Apply ReLU if requested
                data_t out_val = (data_t)sum;
                if (apply_relu && out_val < 0) out_val = 0;

                int out_idx = oc * out_height * out_width + oh * out_width + ow;
                output[out_idx] = out_val;
            }
        }
    }
}
