// ============================================================================
// SQUEEZENET TOP - PROJECT COMPLIANT VERSION
//
// Uses the existing module implementations:
// - conv7x7 from conv3x3.cpp
// - conv1x1 from conv1x1.cpp
// - maxpool from maxpool.cpp
// - fire_module from fire_module.cpp (MODIFIED: uses external DDR squeeze buffer)
// - global_avgpool from maxpool.cpp
//
// KEY REQUIREMENTS MET:
// 1. Uses controller FSM with enable signals (not sequential function calls)
// 2. Calls existing module implementations
// 3. Uses DDR for fire module squeeze buffer (fixes URAM accuracy issue)
// ============================================================================

#include "squeezenet_top.h"
#include "controller.h"
#include "conv1x1.h"
#include "conv3x3.h"
#include "maxpool.h"
#include "fire_module.h"

// ============================================================================
// Top-Level Function - Controller FSM Based Architecture
// ============================================================================
void squeezenet_top(
    data_t *input,
    data_t *output,
    weight_t *weights,
    bias_t *biases,
    data_t *ddr_buffer_a,
    data_t *ddr_buffer_b,
    bool start,
    bool *done
) {
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=150528 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=10 max_write_burst_length=256
    #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem2 depth=740000 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=biases offset=slave bundle=gmem3 depth=3000 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=ddr_buffer_a offset=slave bundle=gmem4 depth=1605632 max_read_burst_length=256 max_write_burst_length=256
    #pragma HLS INTERFACE m_axi port=ddr_buffer_b offset=slave bundle=gmem5 depth=1605632 max_read_burst_length=256 max_write_burst_length=256

    #pragma HLS INTERFACE s_axilite port=input
    #pragma HLS INTERFACE s_axilite port=output
    #pragma HLS INTERFACE s_axilite port=weights
    #pragma HLS INTERFACE s_axilite port=biases
    #pragma HLS INTERFACE s_axilite port=ddr_buffer_a
    #pragma HLS INTERFACE s_axilite port=ddr_buffer_b
    #pragma HLS INTERFACE s_axilite port=start
    #pragma HLS INTERFACE s_axilite port=done
    #pragma HLS INTERFACE s_axilite port=return

    // ========================================================================
    // Controller FSM State
    // ========================================================================
    ControllerState state = STATE_IDLE;
    ModuleEnable enables;

    // ========================================================================
    // Squeeze buffer in DDR - fixes URAM accuracy issue
    //
    // Memory layout of ddr_buffer_a (1,605,632 elements):
    //   [0 - 802,815]:        Feature maps (max 256×56×56 = 802,816)
    //   [1,500,000 - 1,605,631]: Squeeze buffer (max 100,352 elements needed)
    //
    // This region doesn't overlap with feature maps during fire module execution
    // ========================================================================
    data_t* squeeze_buffer = ddr_buffer_a + 1500000;

    // ========================================================================
    // Precomputed weight and bias offsets for each layer
    // ========================================================================
    const int W_CONV1  = 0;
    const int W_FIRE2  = 14112;
    const int W_FIRE3  = 25888;
    const int W_FIRE4  = 38176;
    const int W_FIRE5  = 83232;
    const int W_FIRE6  = 132384;
    const int W_FIRE7  = 236832;
    const int W_FIRE8  = 347424;
    const int W_FIRE9  = 535840;
    const int W_CONV10 = 732448;

    const int B_CONV1  = 0;
    const int B_FIRE2  = 96;
    const int B_FIRE3  = 240;
    const int B_FIRE4  = 384;
    const int B_FIRE5  = 672;
    const int B_FIRE6  = 960;
    const int B_FIRE7  = 1392;
    const int B_FIRE8  = 1824;
    const int B_FIRE9  = 2400;
    const int B_CONV10 = 2976;

    // ========================================================================
    // Main FSM Loop - Processes states until DONE
    // ========================================================================
    FSM_LOOP:
    while (state != STATE_DONE) {
        #pragma HLS LOOP_TRIPCOUNT min=16 max=16

        // Get next state and enable signals from controller
        bool layer_done = true;  // Each layer completes before FSM advances
        state = controller_fsm(state, start, layer_done, &enables);

        // ====================================================================
        // CONV1: 3×224×224 → 96×112×112 (input → ddr_buffer_b)
        // ====================================================================
        if (enables.conv1_en) {
            conv7x7(
                input, ddr_buffer_b,
                weights + W_CONV1, biases + B_CONV1,
                3, 96, 224, 224, 2, 3,
                true, true
            );
        }

        // ====================================================================
        // POOL1: 96×112×112 → 96×56×56 (ddr_buffer_b → ddr_buffer_a)
        // ====================================================================
        if (enables.pool1_en) {
            maxpool(
                ddr_buffer_b, ddr_buffer_a,
                96, 112, 112, 3, 2, true, true
            );
        }

        // ====================================================================
        // FIRE2: 96×56×56 → 128×56×56 (ddr_buffer_a → ddr_buffer_b)
        // ====================================================================
        if (enables.fire2_en) {
            const int sq_w = 16*96;
            const int e1_w = 64*16;
            fire_module(
                ddr_buffer_a, ddr_buffer_b, squeeze_buffer,
                weights + W_FIRE2, biases + B_FIRE2,
                weights + W_FIRE2 + sq_w, biases + B_FIRE2 + 16,
                weights + W_FIRE2 + sq_w + e1_w, biases + B_FIRE2 + 16 + 64,
                96, 16, 64, 64, 56, 56, true
            );
        }

        // ====================================================================
        // FIRE3: 128×56×56 → 128×56×56 (ddr_buffer_b → ddr_buffer_a)
        // ====================================================================
        if (enables.fire3_en) {
            const int sq_w = 16*128;
            const int e1_w = 64*16;
            fire_module(
                ddr_buffer_b, ddr_buffer_a, squeeze_buffer,
                weights + W_FIRE3, biases + B_FIRE3,
                weights + W_FIRE3 + sq_w, biases + B_FIRE3 + 16,
                weights + W_FIRE3 + sq_w + e1_w, biases + B_FIRE3 + 16 + 64,
                128, 16, 64, 64, 56, 56, true
            );
        }

        // ====================================================================
        // FIRE4: 128×56×56 → 256×56×56 (ddr_buffer_a → ddr_buffer_b)
        // ====================================================================
        if (enables.fire4_en) {
            const int sq_w = 32*128;
            const int e1_w = 128*32;
            fire_module(
                ddr_buffer_a, ddr_buffer_b, squeeze_buffer,
                weights + W_FIRE4, biases + B_FIRE4,
                weights + W_FIRE4 + sq_w, biases + B_FIRE4 + 32,
                weights + W_FIRE4 + sq_w + e1_w, biases + B_FIRE4 + 32 + 128,
                128, 32, 128, 128, 56, 56, true
            );
        }

        // ====================================================================
        // POOL2: 256×56×56 → 256×28×28 (ddr_buffer_b → ddr_buffer_a)
        // ====================================================================
        if (enables.pool2_en) {
            maxpool(
                ddr_buffer_b, ddr_buffer_a,
                256, 56, 56, 3, 2, true, true
            );
        }

        // ====================================================================
        // FIRE5: 256×28×28 → 256×28×28 (ddr_buffer_a → ddr_buffer_b)
        // ====================================================================
        if (enables.fire5_en) {
            const int sq_w = 32*256;
            const int e1_w = 128*32;
            fire_module(
                ddr_buffer_a, ddr_buffer_b, squeeze_buffer,
                weights + W_FIRE5, biases + B_FIRE5,
                weights + W_FIRE5 + sq_w, biases + B_FIRE5 + 32,
                weights + W_FIRE5 + sq_w + e1_w, biases + B_FIRE5 + 32 + 128,
                256, 32, 128, 128, 28, 28, true
            );
        }

        // ====================================================================
        // FIRE6: 256×28×28 → 384×28×28 (ddr_buffer_b → ddr_buffer_a)
        // ====================================================================
        if (enables.fire6_en) {
            const int sq_w = 48*256;
            const int e1_w = 192*48;
            fire_module(
                ddr_buffer_b, ddr_buffer_a, squeeze_buffer,
                weights + W_FIRE6, biases + B_FIRE6,
                weights + W_FIRE6 + sq_w, biases + B_FIRE6 + 48,
                weights + W_FIRE6 + sq_w + e1_w, biases + B_FIRE6 + 48 + 192,
                256, 48, 192, 192, 28, 28, true
            );
        }

        // ====================================================================
        // FIRE7: 384×28×28 → 384×28×28 (ddr_buffer_a → ddr_buffer_b)
        // ====================================================================
        if (enables.fire7_en) {
            const int sq_w = 48*384;
            const int e1_w = 192*48;
            fire_module(
                ddr_buffer_a, ddr_buffer_b, squeeze_buffer,
                weights + W_FIRE7, biases + B_FIRE7,
                weights + W_FIRE7 + sq_w, biases + B_FIRE7 + 48,
                weights + W_FIRE7 + sq_w + e1_w, biases + B_FIRE7 + 48 + 192,
                384, 48, 192, 192, 28, 28, true
            );
        }

        // ====================================================================
        // FIRE8: 384×28×28 → 512×28×28 (ddr_buffer_b → ddr_buffer_a)
        // ====================================================================
        if (enables.fire8_en) {
            const int sq_w = 64*384;
            const int e1_w = 256*64;
            fire_module(
                ddr_buffer_b, ddr_buffer_a, squeeze_buffer,
                weights + W_FIRE8, biases + B_FIRE8,
                weights + W_FIRE8 + sq_w, biases + B_FIRE8 + 64,
                weights + W_FIRE8 + sq_w + e1_w, biases + B_FIRE8 + 64 + 256,
                384, 64, 256, 256, 28, 28, true
            );
        }

        // ====================================================================
        // POOL3: 512×28×28 → 512×14×14 (ddr_buffer_a → ddr_buffer_b)
        // ====================================================================
        if (enables.pool3_en) {
            maxpool(
                ddr_buffer_a, ddr_buffer_b,
                512, 28, 28, 3, 2, true, true
            );
        }

        // ====================================================================
        // FIRE9: 512×14×14 → 512×14×14 (ddr_buffer_b → ddr_buffer_a)
        // ====================================================================
        if (enables.fire9_en) {
            const int sq_w = 64*512;
            const int e1_w = 256*64;
            fire_module(
                ddr_buffer_b, ddr_buffer_a, squeeze_buffer,
                weights + W_FIRE9, biases + B_FIRE9,
                weights + W_FIRE9 + sq_w, biases + B_FIRE9 + 64,
                weights + W_FIRE9 + sq_w + e1_w, biases + B_FIRE9 + 64 + 256,
                512, 64, 256, 256, 14, 14, true
            );
        }

        // ====================================================================
        // CONV10: 512×14×14 → 10×14×14 (ddr_buffer_a → ddr_buffer_b)
        // ====================================================================
        if (enables.conv10_en) {
            conv1x1(
                ddr_buffer_a, ddr_buffer_b,
                weights + W_CONV10, biases + B_CONV10,
                512, 10, 14, 14,
                true, true
            );
        }

        // ====================================================================
        // GAP: 10×14×14 → 10 (ddr_buffer_b → output)
        // ====================================================================
        if (enables.gap_en) {
            global_avgpool(
                ddr_buffer_b, output,
                10, 14, 14, true
            );
        }
    }

    *done = true;
}
