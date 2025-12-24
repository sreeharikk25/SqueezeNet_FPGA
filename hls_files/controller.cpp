#include "controller.h"

// ============================================================================
// State Machine Implementation
// ============================================================================

ControllerState controller_fsm(
    ControllerState current_state,
    bool start,
    bool layer_done,
    ModuleEnable *enables
) {
    #pragma HLS INLINE

    // Initialize all enables to false
    enables->conv1_en = false;
    enables->pool1_en = false;
    enables->fire2_en = false;
    enables->fire3_en = false;
    enables->fire4_en = false;
    enables->pool2_en = false;
    enables->fire5_en = false;
    enables->fire6_en = false;
    enables->fire7_en = false;
    enables->fire8_en = false;
    enables->pool3_en = false;
    enables->fire9_en = false;
    enables->conv10_en = false;
    enables->gap_en = false;

    ControllerState next_state = current_state;

    switch (current_state) {
        case STATE_IDLE:
            if (start) next_state = STATE_CONV1;
            break;
        case STATE_CONV1:
            enables->conv1_en = true;
            if (layer_done) next_state = STATE_POOL1;
            break;
        case STATE_POOL1:
            enables->pool1_en = true;
            if (layer_done) next_state = STATE_FIRE2;
            break;
        case STATE_FIRE2:
            enables->fire2_en = true;
            if (layer_done) next_state = STATE_FIRE3;
            break;
        case STATE_FIRE3:
            enables->fire3_en = true;
            if (layer_done) next_state = STATE_FIRE4;
            break;
        case STATE_FIRE4:
            enables->fire4_en = true;
            if (layer_done) next_state = STATE_POOL2;
            break;
        case STATE_POOL2:
            enables->pool2_en = true;
            if (layer_done) next_state = STATE_FIRE5;
            break;
        case STATE_FIRE5:
            enables->fire5_en = true;
            if (layer_done) next_state = STATE_FIRE6;
            break;
        case STATE_FIRE6:
            enables->fire6_en = true;
            if (layer_done) next_state = STATE_FIRE7;
            break;
        case STATE_FIRE7:
            enables->fire7_en = true;
            if (layer_done) next_state = STATE_FIRE8;
            break;
        case STATE_FIRE8:
            enables->fire8_en = true;
            if (layer_done) next_state = STATE_POOL3;
            break;
        case STATE_POOL3:
            enables->pool3_en = true;
            if (layer_done) next_state = STATE_FIRE9;
            break;
        case STATE_FIRE9:
            enables->fire9_en = true;
            if (layer_done) next_state = STATE_CONV10;
            break;
        case STATE_CONV10:
            enables->conv10_en = true;
            if (layer_done) next_state = STATE_GAP;
            break;
        case STATE_GAP:
            enables->gap_en = true;
            if (layer_done) next_state = STATE_DONE;
            break;
        case STATE_DONE:
            next_state = STATE_DONE;
            break;
        default:
            next_state = STATE_IDLE;
            break;
    }

    return next_state;
}

// ============================================================================
// Layer Configuration Generator
// ============================================================================

void get_layer_config(
    int layer_idx,
    LayerConfig *conv_cfg,
    FireConfig *fire_cfg,
    bool *is_fire_module,
    bool *is_pooling
) {
    #pragma HLS INLINE

    *is_fire_module = false;
    *is_pooling = false;

    // Initialize to safe defaults
    conv_cfg->in_channels = 0;
    conv_cfg->out_channels = 0;
    conv_cfg->in_height = 0;
    conv_cfg->in_width = 0;
    conv_cfg->kernel_size = 0;
    conv_cfg->stride = 1;
    conv_cfg->padding = 0;

    switch (layer_idx) {
        case 0:  // conv1
            conv_cfg->in_channels = 3;
            conv_cfg->out_channels = 96;
            conv_cfg->in_height = 224;
            conv_cfg->in_width = 224;
            conv_cfg->kernel_size = 7;
            conv_cfg->stride = 2;
            conv_cfg->padding = 3;
            break;

        case 1:  // pool1
            *is_pooling = true;
            conv_cfg->in_channels = 96;
            conv_cfg->in_height = 112;
            conv_cfg->in_width = 112;
            conv_cfg->kernel_size = 3;
            conv_cfg->stride = 2;
            break;

        case 2:  // fire2
            *is_fire_module = true;
            fire_cfg->in_channels = 96;
            fire_cfg->squeeze_channels = 16;
            fire_cfg->expand1x1_channels = 64;
            fire_cfg->expand3x3_channels = 64;
            fire_cfg->height = 56;
            fire_cfg->width = 56;
            break;

        case 3:  // fire3
            *is_fire_module = true;
            fire_cfg->in_channels = 128;
            fire_cfg->squeeze_channels = 16;
            fire_cfg->expand1x1_channels = 64;
            fire_cfg->expand3x3_channels = 64;
            fire_cfg->height = 56;
            fire_cfg->width = 56;
            break;

        case 4:  // fire4
            *is_fire_module = true;
            fire_cfg->in_channels = 128;
            fire_cfg->squeeze_channels = 32;
            fire_cfg->expand1x1_channels = 128;
            fire_cfg->expand3x3_channels = 128;
            fire_cfg->height = 56;
            fire_cfg->width = 56;
            break;

        case 5:  // pool2
            *is_pooling = true;
            conv_cfg->in_channels = 256;
            conv_cfg->in_height = 56;
            conv_cfg->in_width = 56;
            conv_cfg->kernel_size = 3;
            conv_cfg->stride = 2;
            break;

        case 6:  // fire5
            *is_fire_module = true;
            fire_cfg->in_channels = 256;
            fire_cfg->squeeze_channels = 32;
            fire_cfg->expand1x1_channels = 128;
            fire_cfg->expand3x3_channels = 128;
            fire_cfg->height = 28;
            fire_cfg->width = 28;
            break;

        case 7:  // fire6
            *is_fire_module = true;
            fire_cfg->in_channels = 256;
            fire_cfg->squeeze_channels = 48;
            fire_cfg->expand1x1_channels = 192;
            fire_cfg->expand3x3_channels = 192;
            fire_cfg->height = 28;
            fire_cfg->width = 28;
            break;

        case 8:  // fire7
            *is_fire_module = true;
            fire_cfg->in_channels = 384;
            fire_cfg->squeeze_channels = 48;
            fire_cfg->expand1x1_channels = 192;
            fire_cfg->expand3x3_channels = 192;
            fire_cfg->height = 28;
            fire_cfg->width = 28;
            break;

        case 9:  // fire8
            *is_fire_module = true;
            fire_cfg->in_channels = 384;
            fire_cfg->squeeze_channels = 64;
            fire_cfg->expand1x1_channels = 256;
            fire_cfg->expand3x3_channels = 256;
            fire_cfg->height = 28;
            fire_cfg->width = 28;
            break;

        case 10: // pool3
            *is_pooling = true;
            conv_cfg->in_channels = 512;
            conv_cfg->in_height = 28;
            conv_cfg->in_width = 28;
            conv_cfg->kernel_size = 3;
            conv_cfg->stride = 2;
            break;

        case 11: // fire9
            *is_fire_module = true;
            fire_cfg->in_channels = 512;
            fire_cfg->squeeze_channels = 64;
            fire_cfg->expand1x1_channels = 256;
            fire_cfg->expand3x3_channels = 256;
            fire_cfg->height = 14;
            fire_cfg->width = 14;
            break;

        case 12: // conv10 (classifier)
            conv_cfg->in_channels = 512;
            conv_cfg->out_channels = 10;
            conv_cfg->in_height = 14;
            conv_cfg->in_width = 14;
            conv_cfg->kernel_size = 1;
            conv_cfg->stride = 1;
            conv_cfg->padding = 0;
            break;

        default:
            break;
    }
}

// ============================================================================
// Main Controller Function
// ============================================================================

void squeezenet_controller(
    ModuleEnable *enables,
    MemoryAddr *mem_addr,
    LayerConfig *layer_cfg,
    FireConfig *fire_cfg,
    int *current_layer,
    bool start,
    bool reset,
    bool *done
) {
    #pragma HLS INTERFACE s_axilite port=enables
    #pragma HLS INTERFACE s_axilite port=mem_addr
    #pragma HLS INTERFACE s_axilite port=layer_cfg
    #pragma HLS INTERFACE s_axilite port=fire_cfg
    #pragma HLS INTERFACE s_axilite port=current_layer
    #pragma HLS INTERFACE s_axilite port=start
    #pragma HLS INTERFACE s_axilite port=reset
    #pragma HLS INTERFACE s_axilite port=done
    #pragma HLS INTERFACE s_axilite port=return

    static ControllerState state = STATE_IDLE;
    static int layer_idx = 0;

    if (reset) {
        state = STATE_IDLE;
        layer_idx = 0;
        *done = false;
        return;
    }

    bool layer_done = true;
    state = controller_fsm(state, start, layer_done, enables);

    // Update layer index based on state
    switch (state) {
        case STATE_CONV1:  layer_idx = 0;  break;
        case STATE_POOL1:  layer_idx = 1;  break;
        case STATE_FIRE2:  layer_idx = 2;  break;
        case STATE_FIRE3:  layer_idx = 3;  break;
        case STATE_FIRE4:  layer_idx = 4;  break;
        case STATE_POOL2:  layer_idx = 5;  break;
        case STATE_FIRE5:  layer_idx = 6;  break;
        case STATE_FIRE6:  layer_idx = 7;  break;
        case STATE_FIRE7:  layer_idx = 8;  break;
        case STATE_FIRE8:  layer_idx = 9;  break;
        case STATE_POOL3:  layer_idx = 10; break;
        case STATE_FIRE9:  layer_idx = 11; break;
        case STATE_CONV10: layer_idx = 12; break;
        case STATE_GAP:    layer_idx = 13; break;
        default: break;
    }

    *current_layer = layer_idx;

    bool is_fire, is_pool;
    get_layer_config(layer_idx, layer_cfg, fire_cfg, &is_fire, &is_pool);

    *done = (state == STATE_DONE);
}
