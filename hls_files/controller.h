#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "squeezenet_types.h"  // Contains LayerConfig and FireConfig now

// ============================================================================
// State Machine States
// ============================================================================

typedef enum {
    STATE_IDLE = 0,
    STATE_CONV1,
    STATE_POOL1,
    STATE_FIRE2,
    STATE_FIRE3,
    STATE_FIRE4,
    STATE_POOL2,
    STATE_FIRE5,
    STATE_FIRE6,
    STATE_FIRE7,
    STATE_FIRE8,
    STATE_POOL3,
    STATE_FIRE9,
    STATE_CONV10,
    STATE_GAP,
    STATE_DONE
} ControllerState;

// ============================================================================
// Module Enable Signals
// ============================================================================

typedef struct {
    bool conv1_en;
    bool pool1_en;
    bool fire2_en;
    bool fire3_en;
    bool fire4_en;
    bool pool2_en;
    bool fire5_en;
    bool fire6_en;
    bool fire7_en;
    bool fire8_en;
    bool pool3_en;
    bool fire9_en;
    bool conv10_en;
    bool gap_en;
} ModuleEnable;

// ============================================================================
// Memory Address Structure
// ============================================================================

typedef struct {
    int input_addr;
    int output_addr;
    int weight_addr;
    int bias_addr;
} MemoryAddr;

// ============================================================================
// NOTE: LayerConfig and FireConfig are now in squeezenet_types.h
// to avoid duplicate definitions across multiple headers
// ============================================================================

// ============================================================================
// Function Declarations
// ============================================================================

ControllerState controller_fsm(
    ControllerState current_state,
    bool start,
    bool layer_done,
    ModuleEnable *enables
);

void get_layer_config(
    int layer_idx,
    LayerConfig *conv_cfg,
    FireConfig *fire_cfg,
    bool *is_fire_module,
    bool *is_pooling
);

void squeezenet_controller(
    ModuleEnable *enables,
    MemoryAddr *mem_addr,
    LayerConfig *layer_cfg,
    FireConfig *fire_cfg,
    int *current_layer,
    bool start,
    bool reset,
    bool *done
);

#endif // CONTROLLER_H
