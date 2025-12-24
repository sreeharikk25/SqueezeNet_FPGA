#include "maxpool.h"

// ============================================================================
// Max Pooling Implementation - OPTIMIZED for Q3.4 format
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
) {
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=1605632 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=1605632 max_write_burst_length=256
    #pragma HLS INTERFACE s_axilite port=channels
    #pragma HLS INTERFACE s_axilite port=height
    #pragma HLS INTERFACE s_axilite port=width
    #pragma HLS INTERFACE s_axilite port=kernel_size
    #pragma HLS INTERFACE s_axilite port=stride
    #pragma HLS INTERFACE s_axilite port=ceil_mode
    #pragma HLS INTERFACE s_axilite port=enable
    #pragma HLS INTERFACE s_axilite port=return

    if (!enable) return;

    int out_height, out_width;
    if (ceil_mode) {
        out_height = (height - kernel_size + stride - 1) / stride + 1;
        out_width = (width - kernel_size + stride - 1) / stride + 1;
    } else {
        out_height = (height - kernel_size) / stride + 1;
        out_width = (width - kernel_size) / stride + 1;
    }

    MAXPOOL_C:
    for (int c = 0; c < channels; c++) {
        #pragma HLS LOOP_TRIPCOUNT min=64 max=512

        MAXPOOL_OH:
        for (int oh = 0; oh < out_height; oh++) {
            #pragma HLS LOOP_TRIPCOUNT min=27 max=55

            MAXPOOL_OW:
            for (int ow = 0; ow < out_width; ow++) {
                #pragma HLS LOOP_TRIPCOUNT min=27 max=55

                data_t max_val = (data_t)(-8.0f); // Very negative value for Q3.4

                MAXPOOL_KH:
                for (int kh = 0; kh < kernel_size; kh++) {
                    MAXPOOL_KW:
                    for (int kw = 0; kw < kernel_size; kw++) {
                        #pragma HLS PIPELINE II=1

                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;

                        if (ih < height && iw < width) {
                            int in_idx = c * height * width + ih * width + iw;
                            data_t val = input[in_idx];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                }

                int out_idx = c * out_height * out_width + oh * out_width + ow;
                output[out_idx] = max_val;
            }
        }
    }
}

// ============================================================================
// Global Average Pooling Implementation - OPTIMIZED
// ============================================================================

void global_avgpool(
    data_t *input,
    data_t *output,
    int in_channels,
    int height,
    int width,
    bool enable
) {
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=1605632 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=512 max_write_burst_length=256
    #pragma HLS INTERFACE s_axilite port=in_channels
    #pragma HLS INTERFACE s_axilite port=height
    #pragma HLS INTERFACE s_axilite port=width
    #pragma HLS INTERFACE s_axilite port=enable
    #pragma HLS INTERFACE s_axilite port=return

    if (!enable) return;

    const int HW = height * width;
    acc_t inv_hw = (acc_t)1.0f / (acc_t)HW;

    GAVGPOOL_C:
    for (int c = 0; c < in_channels; c++) {
        #pragma HLS LOOP_TRIPCOUNT min=512 max=512

        acc_t sum = 0;

        GAVGPOOL_H:
        for (int h = 0; h < height; h++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=13

            GAVGPOOL_W:
            for (int w = 0; w < width; w++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=13
                #pragma HLS DEPENDENCE variable=sum inter false

                int idx = c * HW + h * width + w;
                sum += (acc_t)input[idx];
            }
        }

        data_t avg = (data_t)(sum * inv_hw);
        output[c] = avg;
    }
}

// ============================================================================
// Spatial Dropout - OPTIMIZED
// ============================================================================

void spatial_dropout(
    data_t *input,
    data_t *output,
    int channels,
    int height,
    int width,
    float drop_rate,
    bool enable
) {
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=1605632 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=1605632 max_write_burst_length=256
    #pragma HLS INTERFACE s_axilite port=channels
    #pragma HLS INTERFACE s_axilite port=height
    #pragma HLS INTERFACE s_axilite port=width
    #pragma HLS INTERFACE s_axilite port=drop_rate
    #pragma HLS INTERFACE s_axilite port=enable
    #pragma HLS INTERFACE s_axilite port=return

    if (!enable) return;

    const int HW = height * width;

    // During inference, dropout is disabled
    // We just copy input to output with scaling
    acc_t scale = (acc_t)1.0f / ((acc_t)1.0f - (acc_t)drop_rate);

    DROPOUT_C:
    for (int c = 0; c < channels; c++) {
        #pragma HLS LOOP_TRIPCOUNT min=256 max=512

        DROPOUT_HW:
        for (int hw = 0; hw < HW; hw++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=49 max=3136

            int idx = c * HW + hw;
            // For inference, apply scaling factor
            output[idx] = (data_t)((acc_t)input[idx] * scale);
        }
    }
}
