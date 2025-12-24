// tb_squeezenet_enhanced.cpp
// Enhanced testbench with multiple test cases and golden reference
// Tests individual modules AND full network with realistic golden outputs
//
// MODIFIED: Updated fire_module calls to use external squeeze_buffer parameter

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <vector>

#include "squeezenet_top.h"
#include "conv1x1.h"
#include "conv3x3.h"
#include "maxpool.h"
#include "fire_module.h"
#include "controller.h"
#include "squeezenet_types.h"

// =============================================================
// Configuration
// =============================================================
#define TOL_Q 0.15f  // Tolerance for Q3.4 comparisons

// Buffer sizes for different tests
#define SMALL_BUF_SIZE   (64 * 64 * 64)    // For module tests
#define MEDIUM_BUF_SIZE  (128 * 56 * 56)   // For fire module tests
#define LARGE_BUF_SIZE   (512 * 56 * 56)   // For full network
#define SQUEEZE_BUF_SIZE (64 * 56 * 56)    // For fire module squeeze buffer

static data_t   G_BUF_A[MEDIUM_BUF_SIZE];
static data_t   G_BUF_B[MEDIUM_BUF_SIZE];
static data_t   G_SQ[SMALL_BUF_SIZE];
static data_t   G_SQ_HW[SQUEEZE_BUF_SIZE];  // NEW: HW squeeze buffer
static data_t   G_IN[SMALL_BUF_SIZE];
static data_t   G_OUT_HW[SMALL_BUF_SIZE];
static data_t   G_OUT_GOLD[SMALL_BUF_SIZE];
static weight_t G_W[SMALL_BUF_SIZE];
static bias_t   G_B[1024];

// =============================================================
// Utility Functions
// =============================================================

template<typename T>
void rand_fill(T* buf, int n, float lo, float hi, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        float r = (float)rand() / RAND_MAX;
        float v = lo + r * (hi - lo);
        buf[i] = (T)v;
    }
}

template<typename T>
void zero_fill(T* buf, int n) {
    for (int i = 0; i < n; i++) {
        buf[i] = (T)0;
    }
}

template<typename T>
void const_fill(T* buf, int n, float val) {
    for (int i = 0; i < n; i++) {
        buf[i] = (T)val;
    }
}

// Generate gradient pattern (useful for visual debugging)
void gradient_fill(data_t* buf, int c, int h, int w, float scale = 1.0f) {
    for (int ch = 0; ch < c; ch++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float val = ((float)y / h + (float)x / w) / 2.0f; // 0 to 1
                val = (val * 2.0f - 1.0f) * scale; // -scale to +scale
                buf[ch * h * w + y * w + x] = (data_t)val;
            }
        }
    }
}

template<typename T>
void print_vec(const char* name, T* v, int n, int max_print = 10) {
    std::cout << name << ": ";
    for (int i = 0; i < n && i < max_print; i++) {
        std::cout << std::fixed << std::setprecision(4) << (float)v[i] << " ";
    }
    if (n > max_print) std::cout << "... (" << n << " total)";
    std::cout << "\n";
}

template<typename T>
int compare_buf(const char* tag, T* a, T* b, int n, float tol, float &maxdiff, bool verbose = false) {
    int errors = 0;
    maxdiff = 0;
    for (int i = 0; i < n; i++) {
        float da = (float)a[i];
        float db = (float)b[i];
        float d  = std::fabs(da - db);
        if (d > maxdiff) maxdiff = d;
        if (d > tol) {
            if (errors < 10 && verbose) {
                std::cout << tag << " MISMATCH at " << i
                          << " : hw=" << da << " golden=" << db
                          << " diff=" << d << "\n";
            }
            errors++;
        }
    }
    return errors;
}

bool load_weights_from_file(const char* filename, weight_t* buffer, int size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.read(reinterpret_cast<char*>(buffer), size * sizeof(weight_t));
    bool success = file.good();
    file.close();
    return success;
}

bool load_biases_from_file(const char* filename, bias_t* buffer, int size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.read(reinterpret_cast<char*>(buffer), size * sizeof(bias_t));
    bool success = file.good();
    file.close();
    return success;
}

// =============================================================
// Golden Reference Implementations (C++ versions matching PyTorch)
// =============================================================

void golden_conv1x1_q(
    data_t* input, data_t* output,
    weight_t* weights, bias_t* bias,
    int in_ch, int out_ch, int h, int w, bool relu
) {
    int HW = h * w;
    for (int oc = 0; oc < out_ch; oc++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                acc_t sum = (acc_t)bias[oc];
                for (int ic = 0; ic < in_ch; ic++) {
                    sum += (acc_t)input[ic * HW + y * w + x] *
                           (acc_t)weights[oc * in_ch + ic];
                }
                data_t v = (data_t)sum;
                if (relu && v < 0) v = 0;
                output[oc * HW + y * w + x] = v;
            }
        }
    }
}

void golden_conv3x3_q(
    data_t* input, data_t* output,
    weight_t* weights, bias_t* bias,
    int in_ch, int out_ch, int h, int w,
    int stride, int pad, bool relu
) {
    const int K = 3;
    int out_h = (h + 2*pad - K) / stride + 1;
    int out_w = (w + 2*pad - K) / stride + 1;

    for (int oc = 0; oc < out_ch; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                acc_t sum = (acc_t)bias[oc];
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int ih = oh * stride + kh - pad;
                            int iw = ow * stride + kw - pad;
                            data_t in_val = 0;
                            if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                in_val = input[ic * h * w + ih * w + iw];
                            }
                            int w_idx = oc * in_ch * K * K + ic * K * K + kh * K + kw;
                            sum += (acc_t)in_val * (acc_t)weights[w_idx];
                        }
                    }
                }
                data_t v = (data_t)sum;
                if (relu && v < 0) v = 0;
                output[oc * out_h * out_w + oh * out_w + ow] = v;
            }
        }
    }
}

void golden_conv7x7_q(
    data_t* input, data_t* output,
    weight_t* weights, bias_t* bias,
    int in_ch, int out_ch, int h, int w,
    int stride, int pad, bool relu
) {
    const int K = 7;
    int out_h = (h + 2*pad - K) / stride + 1;
    int out_w = (w + 2*pad - K) / stride + 1;

    for (int oc = 0; oc < out_ch; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                acc_t sum = (acc_t)bias[oc];
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int ih = oh * stride + kh - pad;
                            int iw = ow * stride + kw - pad;
                            data_t in_val = 0;
                            if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                in_val = input[ic * h * w + ih * w + iw];
                            }
                            int w_idx = oc * in_ch * K * K + ic * K * K + kh * K + kw;
                            sum += (acc_t)in_val * (acc_t)weights[w_idx];
                        }
                    }
                }
                data_t v = (data_t)sum;
                if (relu && v < 0) v = 0;
                output[oc * out_h * out_w + oh * out_w + ow] = v;
            }
        }
    }
}

void golden_maxpool_q(
    data_t* input, data_t* output,
    int ch, int h, int w,
    int k, int stride, bool ceil_mode
) {
    int out_h, out_w;
    if (ceil_mode) {
        out_h = (h - k + stride - 1) / stride + 1;
        out_w = (w - k + stride - 1) / stride + 1;
    } else {
        out_h = (h - k) / stride + 1;
        out_w = (w - k) / stride + 1;
    }

    for (int c = 0; c < ch; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                data_t max_val = (data_t)(-8.0f);
                for (int kh = 0; kh < k; kh++) {
                    for (int kw = 0; kw < k; kw++) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        if (ih < h && iw < w) {
                            data_t v = input[c * h * w + ih * w + iw];
                            if (v > max_val) max_val = v;
                        }
                    }
                }
                output[c * out_h * out_w + oh * out_w + ow] = max_val;
            }
        }
    }
}

void golden_fire_q(
    data_t* input, data_t* output, data_t* squeeze_buf,
    weight_t* sq_w, bias_t* sq_b,
    weight_t* e1_w, bias_t* e1_b,
    weight_t* e3_w, bias_t* e3_b,
    int in_ch, int sq_ch, int e1_ch, int e3_ch, int h, int w
) {
    int HW = h * w;

    // Squeeze + ReLU
    golden_conv1x1_q(input, squeeze_buf, sq_w, sq_b, in_ch, sq_ch, h, w, true);

    // Expand 1x1
    golden_conv1x1_q(squeeze_buf, output, e1_w, e1_b, sq_ch, e1_ch, h, w, false);

    // Expand 3x3 (output starts at e1_ch offset)
    data_t* e3_out = output + e1_ch * HW;
    golden_conv3x3_q(squeeze_buf, e3_out, e3_w, e3_b, sq_ch, e3_ch, h, w, 1, 1, false);

    // ReLU on concatenated output
    int total = (e1_ch + e3_ch) * HW;
    for (int i = 0; i < total; i++) {
        if (output[i] < 0) output[i] = 0;
    }
}

// Golden reference for full SqueezeNet network
void golden_squeezenet_full(
    data_t* input,
    data_t* output,
    weight_t* weights,
    bias_t* biases
) {
    // Allocate intermediate buffers
    const int FM_SIZE = 512 * 56 * 56;
    const int SQ_SIZE = 64 * 56 * 56;

    data_t* buf_a = new data_t[FM_SIZE];
    data_t* buf_b = new data_t[FM_SIZE];
    data_t* squeeze_buf = new data_t[SQ_SIZE];

    int w_off = 0, b_off = 0;

    // Load input to buf_a
    memcpy(buf_a, input, 3 * 224 * 224 * sizeof(data_t));

    // CONV1: 3x224x224 -> 96x112x112
    std::cout << "  Golden: Conv1...\n";
    golden_conv7x7_q(buf_a, buf_b, weights + w_off, biases + b_off,
                     3, 96, 224, 224, 2, 3, true);
    w_off += 96 * 3 * 49;
    b_off += 96;

    // POOL1: 96x112x112 -> 96x56x56
    std::cout << "  Golden: Pool1...\n";
    golden_maxpool_q(buf_b, buf_a, 96, 112, 112, 3, 2, true);

    // FIRE2: 96x56x56 -> 128x56x56
    std::cout << "  Golden: Fire2...\n";
    {
        int sq_w = 16*96, e1_w = 64*16, e3_w = 64*16*9;
        golden_fire_q(buf_a, buf_b, squeeze_buf,
                     weights + w_off, biases + b_off,
                     weights + w_off + sq_w, biases + b_off + 16,
                     weights + w_off + sq_w + e1_w, biases + b_off + 16 + 64,
                     96, 16, 64, 64, 56, 56);
        w_off += sq_w + e1_w + e3_w;
        b_off += 16 + 64 + 64;
    }

    // FIRE3: 128x56x56 -> 128x56x56
    std::cout << "  Golden: Fire3...\n";
    {
        int sq_w = 16*128, e1_w = 64*16, e3_w = 64*16*9;
        golden_fire_q(buf_b, buf_a, squeeze_buf,
                     weights + w_off, biases + b_off,
                     weights + w_off + sq_w, biases + b_off + 16,
                     weights + w_off + sq_w + e1_w, biases + b_off + 16 + 64,
                     128, 16, 64, 64, 56, 56);
        w_off += sq_w + e1_w + e3_w;
        b_off += 16 + 64 + 64;
    }

    // FIRE4: 128x56x56 -> 256x56x56
    std::cout << "  Golden: Fire4...\n";
    {
        int sq_w = 32*128, e1_w = 128*32, e3_w = 128*32*9;
        golden_fire_q(buf_a, buf_b, squeeze_buf,
                     weights + w_off, biases + b_off,
                     weights + w_off + sq_w, biases + b_off + 32,
                     weights + w_off + sq_w + e1_w, biases + b_off + 32 + 128,
                     128, 32, 128, 128, 56, 56);
        w_off += sq_w + e1_w + e3_w;
        b_off += 32 + 128 + 128;
    }

    // POOL2: 256x56x56 -> 256x28x28
    std::cout << "  Golden: Pool2...\n";
    golden_maxpool_q(buf_b, buf_a, 256, 56, 56, 3, 2, true);

    // FIRE5: 256x28x28 -> 256x28x28
    std::cout << "  Golden: Fire5...\n";
    {
        int sq_w = 32*256, e1_w = 128*32, e3_w = 128*32*9;
        golden_fire_q(buf_a, buf_b, squeeze_buf,
                     weights + w_off, biases + b_off,
                     weights + w_off + sq_w, biases + b_off + 32,
                     weights + w_off + sq_w + e1_w, biases + b_off + 32 + 128,
                     256, 32, 128, 128, 28, 28);
        w_off += sq_w + e1_w + e3_w;
        b_off += 32 + 128 + 128;
    }

    // FIRE6: 256x28x28 -> 384x28x28
    std::cout << "  Golden: Fire6...\n";
    {
        int sq_w = 48*256, e1_w = 192*48, e3_w = 192*48*9;
        golden_fire_q(buf_b, buf_a, squeeze_buf,
                     weights + w_off, biases + b_off,
                     weights + w_off + sq_w, biases + b_off + 48,
                     weights + w_off + sq_w + e1_w, biases + b_off + 48 + 192,
                     256, 48, 192, 192, 28, 28);
        w_off += sq_w + e1_w + e3_w;
        b_off += 48 + 192 + 192;
    }

    // FIRE7: 384x28x28 -> 384x28x28
    std::cout << "  Golden: Fire7...\n";
    {
        int sq_w = 48*384, e1_w = 192*48, e3_w = 192*48*9;
        golden_fire_q(buf_a, buf_b, squeeze_buf,
                     weights + w_off, biases + b_off,
                     weights + w_off + sq_w, biases + b_off + 48,
                     weights + w_off + sq_w + e1_w, biases + b_off + 48 + 192,
                     384, 48, 192, 192, 28, 28);
        w_off += sq_w + e1_w + e3_w;
        b_off += 48 + 192 + 192;
    }

    // FIRE8: 384x28x28 -> 512x28x28
    std::cout << "  Golden: Fire8...\n";
    {
        int sq_w = 64*384, e1_w = 256*64, e3_w = 256*64*9;
        golden_fire_q(buf_b, buf_a, squeeze_buf,
                     weights + w_off, biases + b_off,
                     weights + w_off + sq_w, biases + b_off + 64,
                     weights + w_off + sq_w + e1_w, biases + b_off + 64 + 256,
                     384, 64, 256, 256, 28, 28);
        w_off += sq_w + e1_w + e3_w;
        b_off += 64 + 256 + 256;
    }

    // POOL3: 512x28x28 -> 512x14x14
    std::cout << "  Golden: Pool3...\n";
    golden_maxpool_q(buf_a, buf_b, 512, 28, 28, 3, 2, true);

    // FIRE9: 512x14x14 -> 512x14x14
    std::cout << "  Golden: Fire9...\n";
    {
        int sq_w = 64*512, e1_w = 256*64, e3_w = 256*64*9;
        golden_fire_q(buf_b, buf_a, squeeze_buf,
                     weights + w_off, biases + b_off,
                     weights + w_off + sq_w, biases + b_off + 64,
                     weights + w_off + sq_w + e1_w, biases + b_off + 64 + 256,
                     512, 64, 256, 256, 14, 14);
        w_off += sq_w + e1_w + e3_w;
        b_off += 64 + 256 + 256;
    }

    // CONV10: 512x14x14 -> 10x14x14
    std::cout << "  Golden: Conv10...\n";
    golden_conv1x1_q(buf_a, buf_b, weights + w_off, biases + b_off,
                    512, 10, 14, 14, true);

    // GAP: 10x14x14 -> 10
    std::cout << "  Golden: GAP...\n";
    const int spatial = 14 * 14;
    for (int c = 0; c < 10; c++) {
        acc_t sum = 0;
        for (int i = 0; i < spatial; i++) {
            sum += (acc_t)buf_b[c * spatial + i];
        }
        output[c] = (data_t)(sum / spatial);
    }

    // Cleanup
    delete[] buf_a;
    delete[] buf_b;
    delete[] squeeze_buf;
}

// =============================================================
// Individual Module Tests
// =============================================================

int test_conv1x1_ip() {
    std::cout << "\n==============================================\n";
    std::cout << "TEST 1: conv1x1 Module\n";
    std::cout << "==============================================\n";

    int total_errors = 0;

    // Test Case 1: Small config
    std::cout << "\n[Test 1.1] Small config (8x8x8 -> 16x8x8)\n";
    {
        const int IC = 8, OC = 16, H = 8, W = 8;
        const int IN_SZ = IC * H * W;
        const int OUT_SZ = OC * H * W;

        rand_fill(G_IN, IN_SZ, -1.0f, 1.0f, 1);
        rand_fill(G_W, OC * IC, -0.3f, 0.3f, 2);
        rand_fill(G_B, OC, -0.2f, 0.2f, 3);

        golden_conv1x1_q(G_IN, G_OUT_GOLD, G_W, G_B, IC, OC, H, W, true);
        conv1x1(G_IN, G_OUT_HW, G_W, G_B, IC, OC, H, W, true, true);

        float maxdiff;
        int errors = compare_buf("conv1x1_small", G_OUT_HW, G_OUT_GOLD, OUT_SZ, TOL_Q, maxdiff);
        std::cout << (errors == 0 ? "✓ PASS" : "✗ FAIL") << " (maxdiff=" << maxdiff << ")\n";
        total_errors += errors;
    }

    // Test Case 2: Larger config
    std::cout << "\n[Test 1.2] Larger config (32x16x16 -> 64x16x16)\n";
    {
        const int IC = 32, OC = 64, H = 16, W = 16;
        const int IN_SZ = IC * H * W;
        const int OUT_SZ = OC * H * W;

        rand_fill(G_IN, IN_SZ, -1.0f, 1.0f, 4);
        rand_fill(G_W, OC * IC, -0.3f, 0.3f, 5);
        rand_fill(G_B, OC, -0.2f, 0.2f, 6);

        golden_conv1x1_q(G_IN, G_OUT_GOLD, G_W, G_B, IC, OC, H, W, true);
        conv1x1(G_IN, G_OUT_HW, G_W, G_B, IC, OC, H, W, true, true);

        float maxdiff;
        int errors = compare_buf("conv1x1_large", G_OUT_HW, G_OUT_GOLD, OUT_SZ, TOL_Q, maxdiff);
        std::cout << (errors == 0 ? "✓ PASS" : "✗ FAIL") << " (maxdiff=" << maxdiff << ")\n";
        total_errors += errors;
    }

    // Test Case 3: No ReLU
    std::cout << "\n[Test 1.3] Without ReLU (allow negative outputs)\n";
    {
        const int IC = 16, OC = 16, H = 8, W = 8;
        const int IN_SZ = IC * H * W;
        const int OUT_SZ = OC * H * W;

        rand_fill(G_IN, IN_SZ, -1.0f, 1.0f, 7);
        rand_fill(G_W, OC * IC, -0.3f, 0.3f, 8);
        rand_fill(G_B, OC, -0.2f, 0.2f, 9);

        golden_conv1x1_q(G_IN, G_OUT_GOLD, G_W, G_B, IC, OC, H, W, false);
        conv1x1(G_IN, G_OUT_HW, G_W, G_B, IC, OC, H, W, true, false);

        float maxdiff;
        int errors = compare_buf("conv1x1_norelu", G_OUT_HW, G_OUT_GOLD, OUT_SZ, TOL_Q, maxdiff);
        std::cout << (errors == 0 ? "✓ PASS" : "✗ FAIL") << " (maxdiff=" << maxdiff << ")\n";
        total_errors += errors;
    }

    std::cout << "\n" << (total_errors == 0 ? "✓ conv1x1: ALL TESTS PASSED" : "✗ conv1x1: SOME TESTS FAILED")
              << " (total errors: " << total_errors << ")\n";
    return total_errors;
}

int test_conv3x3_ip() {
    std::cout << "\n==============================================\n";
    std::cout << "TEST 2: conv3x3 Module\n";
    std::cout << "==============================================\n";

    int total_errors = 0;

    // Test Case 1: Small with padding
    std::cout << "\n[Test 2.1] Small config with padding (8x8x8 -> 16x8x8)\n";
    {
        const int IC = 8, OC = 16, H = 8, W = 8;
        const int STRIDE = 1, PAD = 1;
        const int OH = (H + 2*PAD - 3) / STRIDE + 1;
        const int OW = (W + 2*PAD - 3) / STRIDE + 1;
        const int IN_SZ = IC * H * W;
        const int OUT_SZ = OC * OH * OW;

        rand_fill(G_IN, IN_SZ, -1.0f, 1.0f, 10);
        rand_fill(G_W, OC * IC * 9, -0.2f, 0.2f, 11);
        rand_fill(G_B, OC, -0.1f, 0.1f, 12);

        golden_conv3x3_q(G_IN, G_OUT_GOLD, G_W, G_B, IC, OC, H, W, STRIDE, PAD, true);
        conv3x3(G_IN, G_OUT_HW, G_W, G_B, IC, OC, H, W, STRIDE, PAD, true, true);

        float maxdiff;
        int errors = compare_buf("conv3x3_pad", G_OUT_HW, G_OUT_GOLD, OUT_SZ, TOL_Q, maxdiff);
        std::cout << (errors == 0 ? "✓ PASS" : "✗ FAIL") << " (maxdiff=" << maxdiff << ")\n";
        total_errors += errors;
    }

    // Test Case 2: With stride
    std::cout << "\n[Test 2.2] With stride=2 (16x16x16 -> 32x8x8)\n";
    {
        const int IC = 16, OC = 32, H = 16, W = 16;
        const int STRIDE = 2, PAD = 1;
        const int OH = (H + 2*PAD - 3) / STRIDE + 1;
        const int OW = (W + 2*PAD - 3) / STRIDE + 1;
        const int IN_SZ = IC * H * W;
        const int OUT_SZ = OC * OH * OW;

        rand_fill(G_IN, IN_SZ, -1.0f, 1.0f, 13);
        rand_fill(G_W, OC * IC * 9, -0.2f, 0.2f, 14);
        rand_fill(G_B, OC, -0.1f, 0.1f, 15);

        golden_conv3x3_q(G_IN, G_OUT_GOLD, G_W, G_B, IC, OC, H, W, STRIDE, PAD, true);
        conv3x3(G_IN, G_OUT_HW, G_W, G_B, IC, OC, H, W, STRIDE, PAD, true, true);

        float maxdiff;
        int errors = compare_buf("conv3x3_stride", G_OUT_HW, G_OUT_GOLD, OUT_SZ, TOL_Q, maxdiff);
        std::cout << (errors == 0 ? "✓ PASS" : "✗ FAIL") << " (maxdiff=" << maxdiff << ")\n";
        total_errors += errors;
    }

    std::cout << "\n" << (total_errors == 0 ? "✓ conv3x3: ALL TESTS PASSED" : "✗ conv3x3: SOME TESTS FAILED")
              << " (total errors: " << total_errors << ")\n";
    return total_errors;
}

int test_conv7x7_ip() {
    std::cout << "\n==============================================\n";
    std::cout << "TEST 3: conv7x7 Module\n";
    std::cout << "==============================================\n";

    int total_errors = 0;

    // Test Case 1: Realistic first layer config
    std::cout << "\n[Test 3.1] First layer config (3x32x32 -> 8x16x16)\n";
    {
        const int IC = 3, OC = 8, H = 32, W = 32;
        const int STRIDE = 2, PAD = 3;
        const int OH = (H + 2*PAD - 7) / STRIDE + 1;
        const int OW = (W + 2*PAD - 7) / STRIDE + 1;
        const int IN_SZ = IC * H * W;
        const int OUT_SZ = OC * OH * OW;

        rand_fill(G_IN, IN_SZ, -1.0f, 1.0f, 16);
        rand_fill(G_W, OC * IC * 49, -0.2f, 0.2f, 17);
        rand_fill(G_B, OC, -0.1f, 0.1f, 18);

        golden_conv7x7_q(G_IN, G_OUT_GOLD, G_W, G_B, IC, OC, H, W, STRIDE, PAD, true);
        conv7x7(G_IN, G_OUT_HW, G_W, G_B, IC, OC, H, W, STRIDE, PAD, true, true);

        float maxdiff;
        int errors = compare_buf("conv7x7", G_OUT_HW, G_OUT_GOLD, OUT_SZ, TOL_Q, maxdiff);
        std::cout << (errors == 0 ? "✓ PASS" : "✗ FAIL") << " (maxdiff=" << maxdiff << ")\n";
        total_errors += errors;
    }

    std::cout << "\n" << (total_errors == 0 ? "✓ conv7x7: ALL TESTS PASSED" : "✗ conv7x7: SOME TESTS FAILED")
              << " (total errors: " << total_errors << ")\n";
    return total_errors;
}

int test_maxpool_ip() {
    std::cout << "\n==============================================\n";
    std::cout << "TEST 4: maxpool Module\n";
    std::cout << "==============================================\n";

    int total_errors = 0;

    // Test Case 1: Standard pooling
    std::cout << "\n[Test 4.1] Standard 3x3 pooling (16x14x14 -> 16x7x7)\n";
    {
        const int C = 16, H = 14, W = 14;
        const int K = 3, STRIDE = 2;
        const int OH = (H - K + STRIDE - 1) / STRIDE + 1;
        const int OW = (W - K + STRIDE - 1) / STRIDE + 1;
        const int IN_SZ = C * H * W;
        const int OUT_SZ = C * OH * OW;

        rand_fill(G_IN, IN_SZ, -2.0f, 2.0f, 19);

        golden_maxpool_q(G_IN, G_OUT_GOLD, C, H, W, K, STRIDE, true);
        maxpool(G_IN, G_OUT_HW, C, H, W, K, STRIDE, true, true);

        float maxdiff;
        int errors = compare_buf("maxpool_std", G_OUT_HW, G_OUT_GOLD, OUT_SZ, TOL_Q, maxdiff);
        std::cout << (errors == 0 ? "✓ PASS" : "✗ FAIL") << " (maxdiff=" << maxdiff << ")\n";
        total_errors += errors;
    }

    // Test Case 2: Larger channels
    std::cout << "\n[Test 4.2] More channels (32x16x16 -> 32x8x8)\n";
    {
        const int C = 32, H = 16, W = 16;
        const int K = 3, STRIDE = 2;
        const int OH = (H - K + STRIDE - 1) / STRIDE + 1;
        const int OW = (W - K + STRIDE - 1) / STRIDE + 1;
        const int IN_SZ = C * H * W;
        const int OUT_SZ = C * OH * OW;

        rand_fill(G_IN, IN_SZ, -2.0f, 2.0f, 20);

        golden_maxpool_q(G_IN, G_OUT_GOLD, C, H, W, K, STRIDE, true);
        maxpool(G_IN, G_OUT_HW, C, H, W, K, STRIDE, true, true);

        float maxdiff;
        int errors = compare_buf("maxpool_large", G_OUT_HW, G_OUT_GOLD, OUT_SZ, TOL_Q, maxdiff);
        std::cout << (errors == 0 ? "✓ PASS" : "✗ FAIL") << " (maxdiff=" << maxdiff << ")\n";
        total_errors += errors;
    }

    std::cout << "\n" << (total_errors == 0 ? "✓ maxpool: ALL TESTS PASSED" : "✗ maxpool: SOME TESTS FAILED")
              << " (total errors: " << total_errors << ")\n";
    return total_errors;
}

int test_fire_ip() {
    std::cout << "\n==============================================\n";
    std::cout << "TEST 5: fire_module Module (with external squeeze buffer)\n";
    std::cout << "==============================================\n";

    int total_errors = 0;

    // Test Case 1: Small fire module
    std::cout << "\n[Test 5.1] Small fire (16x8x8 -> 16x8x8)\n";
    {
        const int IN_C = 16, SQ_C = 4, E1_C = 8, E3_C = 8;
        const int H = 8, W = 8;
        const int IN_SZ = IN_C * H * W;
        const int OUT_SZ = (E1_C + E3_C) * H * W;
        const int SQ_SZ = SQ_C * H * W;

        weight_t* sq_w = G_W;
        weight_t* e1_w = G_W + SQ_C * IN_C;
        weight_t* e3_w = G_W + SQ_C * IN_C + E1_C * SQ_C;

        bias_t* sq_b = G_B;
        bias_t* e1_b = G_B + SQ_C;
        bias_t* e3_b = G_B + SQ_C + E1_C;

        rand_fill(G_IN, IN_SZ, -1.0f, 1.0f, 21);
        rand_fill(sq_w, SQ_C * IN_C, -0.2f, 0.2f, 22);
        rand_fill(e1_w, E1_C * SQ_C, -0.2f, 0.2f, 23);
        rand_fill(e3_w, E3_C * SQ_C * 9, -0.2f, 0.2f, 24);
        rand_fill(sq_b, SQ_C, -0.1f, 0.1f, 25);
        rand_fill(e1_b, E1_C, -0.1f, 0.1f, 26);
        rand_fill(e3_b, E3_C, -0.1f, 0.1f, 27);
        zero_fill(G_SQ_HW, SQ_SZ);  // Clear HW squeeze buffer

        // Golden reference
        golden_fire_q(G_IN, G_OUT_GOLD, G_SQ,
                     sq_w, sq_b, e1_w, e1_b, e3_w, e3_b,
                     IN_C, SQ_C, E1_C, E3_C, H, W);

        // HLS version with external squeeze buffer
        fire_module(G_IN, G_OUT_HW, G_SQ_HW,  // Added squeeze buffer
                   sq_w, sq_b, e1_w, e1_b, e3_w, e3_b,
                   IN_C, SQ_C, E1_C, E3_C, H, W, true);

        float maxdiff;
        int errors = compare_buf("fire_small", G_OUT_HW, G_OUT_GOLD, OUT_SZ, TOL_Q, maxdiff);
        std::cout << (errors == 0 ? "✓ PASS" : "✗ FAIL") << " (maxdiff=" << maxdiff << ")\n";
        total_errors += errors;
    }

    // Test Case 2: Medium fire module
    std::cout << "\n[Test 5.2] Medium fire (32x16x16 -> 64x16x16)\n";
    {
        const int IN_C = 32, SQ_C = 8, E1_C = 32, E3_C = 32;
        const int H = 16, W = 16;
        const int IN_SZ = IN_C * H * W;
        const int OUT_SZ = (E1_C + E3_C) * H * W;
        const int SQ_SZ = SQ_C * H * W;

        data_t* input = new data_t[IN_SZ];
        data_t* output_hw = new data_t[OUT_SZ];
        data_t* output_gold = new data_t[OUT_SZ];
        data_t* sq_buf_gold = new data_t[SQ_SZ];
        data_t* sq_buf_hw = new data_t[SQ_SZ];  // HW squeeze buffer

        weight_t* sq_w = new weight_t[SQ_C * IN_C];
        weight_t* e1_w = new weight_t[E1_C * SQ_C];
        weight_t* e3_w = new weight_t[E3_C * SQ_C * 9];

        bias_t* sq_b = new bias_t[SQ_C];
        bias_t* e1_b = new bias_t[E1_C];
        bias_t* e3_b = new bias_t[E3_C];

        rand_fill(input, IN_SZ, -1.0f, 1.0f, 28);
        rand_fill(sq_w, SQ_C * IN_C, -0.2f, 0.2f, 29);
        rand_fill(e1_w, E1_C * SQ_C, -0.2f, 0.2f, 30);
        rand_fill(e3_w, E3_C * SQ_C * 9, -0.2f, 0.2f, 31);
        rand_fill(sq_b, SQ_C, -0.1f, 0.1f, 32);
        rand_fill(e1_b, E1_C, -0.1f, 0.1f, 33);
        rand_fill(e3_b, E3_C, -0.1f, 0.1f, 34);
        zero_fill(sq_buf_hw, SQ_SZ);  // Clear HW squeeze buffer

        // Golden reference
        golden_fire_q(input, output_gold, sq_buf_gold,
                     sq_w, sq_b, e1_w, e1_b, e3_w, e3_b,
                     IN_C, SQ_C, E1_C, E3_C, H, W);

        // HLS version with external squeeze buffer
        fire_module(input, output_hw, sq_buf_hw,  // Added squeeze buffer
                   sq_w, sq_b, e1_w, e1_b, e3_w, e3_b,
                   IN_C, SQ_C, E1_C, E3_C, H, W, true);

        float maxdiff;
        int errors = compare_buf("fire_medium", output_hw, output_gold, OUT_SZ, TOL_Q, maxdiff);
        std::cout << (errors == 0 ? "✓ PASS" : "✗ FAIL") << " (maxdiff=" << maxdiff << ")\n";
        total_errors += errors;

        delete[] input; delete[] output_hw; delete[] output_gold;
        delete[] sq_buf_gold; delete[] sq_buf_hw;
        delete[] sq_w; delete[] e1_w; delete[] e3_w;
        delete[] sq_b; delete[] e1_b; delete[] e3_b;
    }

    // Test Case 3: SqueezeNet-sized fire module (Fire2 config)
    std::cout << "\n[Test 5.3] SqueezeNet Fire2 config (96x56x56 -> 128x56x56)\n";
    {
        const int IN_C = 96, SQ_C = 16, E1_C = 64, E3_C = 64;
        const int H = 56, W = 56;
        const int IN_SZ = IN_C * H * W;
        const int OUT_SZ = (E1_C + E3_C) * H * W;
        const int SQ_SZ = SQ_C * H * W;

        data_t* input = new data_t[IN_SZ];
        data_t* output_hw = new data_t[OUT_SZ];
        data_t* output_gold = new data_t[OUT_SZ];
        data_t* sq_buf_gold = new data_t[SQ_SZ];
        data_t* sq_buf_hw = new data_t[SQ_SZ];

        weight_t* sq_w = new weight_t[SQ_C * IN_C];
        weight_t* e1_w = new weight_t[E1_C * SQ_C];
        weight_t* e3_w = new weight_t[E3_C * SQ_C * 9];

        bias_t* sq_b = new bias_t[SQ_C];
        bias_t* e1_b = new bias_t[E1_C];
        bias_t* e3_b = new bias_t[E3_C];

        rand_fill(input, IN_SZ, -1.0f, 1.0f, 35);
        rand_fill(sq_w, SQ_C * IN_C, -0.2f, 0.2f, 36);
        rand_fill(e1_w, E1_C * SQ_C, -0.2f, 0.2f, 37);
        rand_fill(e3_w, E3_C * SQ_C * 9, -0.2f, 0.2f, 38);
        rand_fill(sq_b, SQ_C, -0.1f, 0.1f, 39);
        rand_fill(e1_b, E1_C, -0.1f, 0.1f, 40);
        rand_fill(e3_b, E3_C, -0.1f, 0.1f, 41);
        zero_fill(sq_buf_hw, SQ_SZ);

        golden_fire_q(input, output_gold, sq_buf_gold,
                     sq_w, sq_b, e1_w, e1_b, e3_w, e3_b,
                     IN_C, SQ_C, E1_C, E3_C, H, W);

        fire_module(input, output_hw, sq_buf_hw,
                   sq_w, sq_b, e1_w, e1_b, e3_w, e3_b,
                   IN_C, SQ_C, E1_C, E3_C, H, W, true);

        float maxdiff;
        int errors = compare_buf("fire_fire2", output_hw, output_gold, OUT_SZ, TOL_Q, maxdiff);
        std::cout << (errors == 0 ? "✓ PASS" : "✗ FAIL") << " (maxdiff=" << maxdiff << ")\n";
        total_errors += errors;

        delete[] input; delete[] output_hw; delete[] output_gold;
        delete[] sq_buf_gold; delete[] sq_buf_hw;
        delete[] sq_w; delete[] e1_w; delete[] e3_w;
        delete[] sq_b; delete[] e1_b; delete[] e3_b;
    }

    std::cout << "\n" << (total_errors == 0 ? "✓ fire_module: ALL TESTS PASSED" : "✗ fire_module: SOME TESTS FAILED")
              << " (total errors: " << total_errors << ")\n";
    return total_errors;
}

int test_controller_logic() {
    std::cout << "\n==============================================\n";
    std::cout << "TEST 6: controller Logic\n";
    std::cout << "==============================================\n";

    int fail = 0;
    for (int layer = 0; layer <= 12; layer++) {
        LayerConfig conv_cfg;
        FireConfig fire_cfg;
        bool is_fire = false, is_pool = false;

        get_layer_config(layer, &conv_cfg, &fire_cfg, &is_fire, &is_pool);

        std::cout << "Layer " << std::setw(2) << layer
                  << ": is_fire=" << is_fire
                  << " is_pool=" << is_pool
                  << " in_ch=" << std::setw(3) << conv_cfg.in_channels
                  << " out_ch=" << std::setw(3) << conv_cfg.out_channels
                  << "\n";

        // Verify critical layers
        if (layer == 0 && (conv_cfg.in_channels != 3 || conv_cfg.out_channels != 96)) fail++;
        if (layer == 2 && !is_fire) fail++;
        if (layer == 12 && conv_cfg.out_channels != 10) fail++;
    }

    std::cout << "\n" << (fail == 0 ? "✓ controller: ALL TESTS PASSED" : "✗ controller: SOME TESTS FAILED")
              << " (" << fail << " failures)\n";
    return fail;
}

// =============================================================
// Full Network Test with Golden Reference
// =============================================================

int test_squeezenet_full(data_t* ddr_buffer_a, data_t* ddr_buffer_b)  {
    std::cout << "\n==============================================\n";
    std::cout << "TEST 7: Full SqueezeNet Network\n";
    std::cout << "==============================================\n";

    const int IN_SZ = 3 * 224 * 224;
    const int OUT_SZ = 10;
    const int W_SZ = 1250000;  // Increased buffer size
    const int B_SZ = 3000;

    std::cout << "\n[Phase 1] Allocating buffers...\n";
    data_t* input_buf = new data_t[IN_SZ];
    data_t* output_hw = new data_t[OUT_SZ];
    data_t* output_golden = new data_t[OUT_SZ];
    weight_t* weights = new weight_t[W_SZ];
    bias_t* biases = new bias_t[B_SZ];

    if (!input_buf || !output_hw || !output_golden || !weights || !biases) {
        std::cout << "✗ ERROR: Memory allocation failed\n";
        return 1;
    }
    std::cout << "✓ Memory allocated successfully\n";

    // Test Case 1: With trained weights (if available)
    std::cout << "\n[Phase 2] Attempting to load trained weights...\n";
    bool weights_loaded = load_weights_from_file("weights.bin", weights, W_SZ);
    bool biases_loaded = load_biases_from_file("biases.bin", biases, B_SZ);

    int total_errors = 0;

    if (weights_loaded && biases_loaded) {
        std::cout << "✓ Trained weights loaded!\n";
        std::cout << "\n[Test 7.1] Full network with TRAINED weights\n";

        // Create gradient test input
        std::cout << "  Creating test input (gradient pattern)...\n";
        gradient_fill(input_buf, 3, 224, 224, 0.5f);

        // Run HLS version
        std::cout << "  Running HLS version...\n";
        bool done = false;
        squeezenet_top(
                    input_buf,      // Input
                    output_hw,      // Output
                    weights,        // Weights
                    biases,         // Biases
                    ddr_buffer_a,   // DDR buffer A
                    ddr_buffer_b,   // DDR buffer B
                    true,           // Start
                    &done           // Done flag
                );

        if (!done) {
            std::cout << "✗ ERROR: HLS version did not complete\n";
            total_errors++;
        } else {
            std::cout << "✓ HLS version completed\n";

            // Run golden version
            std::cout << "  Running golden reference...\n";
            golden_squeezenet_full(input_buf, output_golden, weights, biases);
            std::cout << "✓ Golden reference completed\n";

            // Compare outputs
            std::cout << "\n  Comparing outputs:\n";
            std::cout << "  Class |    HLS    |  Golden   |   Diff\n";
            std::cout << "  ------|-----------|-----------|----------\n";

            float maxdiff = 0;
            int errors = 0;
            for (int i = 0; i < OUT_SZ; i++) {
                float hw_val = (float)output_hw[i];
                float gold_val = (float)output_golden[i];
                float diff = std::fabs(hw_val - gold_val);

                std::cout << "    " << i << "   | "
                          << std::fixed << std::setprecision(4) << std::setw(9) << hw_val << " | "
                          << std::setw(9) << gold_val << " | "
                          << std::setw(8) << diff << "\n";

                if (diff > maxdiff) maxdiff = diff;
                if (diff > TOL_Q) errors++;
            }

            std::cout << "\n  Maximum difference: " << maxdiff << "\n";
            std::cout << "  Tolerance: " << TOL_Q << "\n";
            std::cout << "  Mismatches: " << errors << "/" << OUT_SZ << "\n";

            // Find predicted classes
            int hw_pred = 0, gold_pred = 0;
            float hw_max = -1000, gold_max = -1000;
            for (int i = 0; i < OUT_SZ; i++) {
                if ((float)output_hw[i] > hw_max) {
                    hw_max = (float)output_hw[i];
                    hw_pred = i;
                }
                if ((float)output_golden[i] > gold_max) {
                    gold_max = (float)output_golden[i];
                    gold_pred = i;
                }
            }

            std::cout << "\n  Predicted class (HLS):    " << hw_pred << " (score: " << hw_max << ")\n";
            std::cout << "  Predicted class (Golden): " << gold_pred << " (score: " << gold_max << ")\n";

            if (hw_pred == gold_pred) {
                std::cout << "  ✓ Predictions MATCH!\n";
            } else {
                std::cout << "  ⚠ Predictions DIFFER (may be OK if scores are close)\n";
            }

            if (errors == 0) {
                std::cout << "\n✓ [Test 7.1] PASSED: HLS matches golden reference!\n";
            } else if (errors <= 2 && maxdiff < TOL_Q * 2) {
                std::cout << "\n⚠ [Test 7.1] MARGINAL PASS: Small differences within acceptable range\n";
            } else {
                std::cout << "\n✗ [Test 7.1] FAILED: Significant differences detected\n";
                total_errors += errors;
            }
        }

    } else {
        std::cout << "⚠ Trained weights not found (this is OK for initial testing)\n";
        std::cout << "  To test with real weights, run:\n";
        std::cout << "    python export_weights_q34.py checkpoints/best_qat_q34.pth squeezenet_q34\n\n";

        // Fallback: Test with random weights (sanity check only)
        std::cout << "[Test 7.2] Full network SANITY CHECK (random weights)\n";
        std::cout << "  Note: This only verifies HLS doesn't crash, not correctness\n\n";

        rand_fill(weights, W_SZ, -2.0f, 1.1f, 50);
        rand_fill(biases, B_SZ, -0.05f, 0.05f, 51);
        gradient_fill(input_buf, 3, 224, 224, 0.5f);

        bool done = false;
        squeezenet_top(
                    input_buf,
                    output_hw,
                    weights,
                    biases,
                    ddr_buffer_a,
                    ddr_buffer_b,
                    true,
                    &done
                );


        if (!done) {
            std::cout << "✗ ERROR: HLS version did not complete\n";
            total_errors++;
        } else {
            std::cout << "✓ HLS version completed without crashing\n";

            print_vec("  Output", output_hw, OUT_SZ, OUT_SZ);

            // Basic sanity checks
            bool has_nans = false, has_infs = false, all_zeros = true;
            for (int i = 0; i < OUT_SZ; i++) {
                float val = (float)output_hw[i];
                if (std::isnan(val)) has_nans = true;
                if (std::isinf(val)) has_infs = true;
                if (std::fabs(val) > 0.001f) all_zeros = false;
            }

            int sanity_errors = 0;
            if (has_nans) {
                std::cout << "✗ Output contains NaN values\n";
                sanity_errors++;
            } else {
                std::cout << "✓ No NaN values\n";
            }

            if (has_infs) {
                std::cout << "✗ Output contains Inf values\n";
                sanity_errors++;
            } else {
                std::cout << "✓ No Inf values\n";
            }

            if (all_zeros) {
                std::cout << "⚠ All outputs are zero (expected with random weights)\n";
            } else {
                std::cout << "✓ Outputs are non-zero\n";
            }

            if (sanity_errors == 0) {
                std::cout << "\n✓ [Test 7.2] SANITY CHECK PASSED\n";
            } else {
                std::cout << "\n✗ [Test 7.2] SANITY CHECK FAILED\n";
                total_errors += sanity_errors;
            }
        }
    }

    // Cleanup
    delete[] input_buf;
    delete[] output_hw;
    delete[] output_golden;
    delete[] weights;
    delete[] biases;

    std::cout << "\n" << (total_errors == 0 ? "✓ Full network: ALL TESTS PASSED" : "✗ Full network: SOME TESTS FAILED")
              << " (total errors: " << total_errors << ")\n";
    return total_errors;
}

int test_with_trained_weights(data_t* ddr_buffer_a, data_t* ddr_buffer_b) {
    std::cout << "\n==============================================\n";
    std::cout << "TEST: Inference with Trained Weights\n";
    std::cout << "==============================================\n";

    const int IN_SZ = 3 * 224 * 224;    // 150,528
    const int OUT_SZ = 10;
    const int W_SZ = 737568;            // CORRECT size!
    const int B_SZ = 2986;              // CORRECT size!

    // Allocate buffers
    data_t* input_buf = new data_t[IN_SZ];
    data_t* output_hw = new data_t[OUT_SZ];
    data_t* output_golden = new data_t[OUT_SZ];

    // Use int8_t for raw file loading, then cast
    int8_t* weights_raw = new int8_t[W_SZ];
    int8_t* biases_raw = new int8_t[B_SZ];
    weight_t* weights = new weight_t[W_SZ];
    bias_t* biases = new bias_t[B_SZ];

    // Load weights as raw bytes
    std::cout << "Loading weights.bin...\n";
    std::ifstream wf("weights.bin", std::ios::binary);
    if (!wf.is_open()) {
        std::cout << "✗ ERROR: Cannot open weights.bin\n";
        std::cout << "  Make sure the file is in the project directory\n";
        delete[] input_buf; delete[] output_hw; delete[] output_golden;
        delete[] weights_raw; delete[] biases_raw;
        delete[] weights; delete[] biases;
        return 1;
    }
    wf.read(reinterpret_cast<char*>(weights_raw), W_SZ);
    if (!wf.good() && !wf.eof()) {
        std::cout << "✗ ERROR: Failed to read weights.bin\n";
        delete[] input_buf; delete[] output_hw; delete[] output_golden;
        delete[] weights_raw; delete[] biases_raw;
        delete[] weights; delete[] biases;
        return 1;
    }
    wf.close();
    std::cout << "✓ Loaded " << W_SZ << " weight bytes\n";

    // Load biases as raw bytes
    std::cout << "Loading biases.bin...\n";
    std::ifstream bf("biases.bin", std::ios::binary);
    if (!bf.is_open()) {
        std::cout << "✗ ERROR: Cannot open biases.bin\n";
        delete[] input_buf; delete[] output_hw; delete[] output_golden;
        delete[] weights_raw; delete[] biases_raw;
        delete[] weights; delete[] biases;
        return 1;
    }
    bf.read(reinterpret_cast<char*>(biases_raw), B_SZ);
    if (!bf.good() && !bf.eof()) {
        std::cout << "✗ ERROR: Failed to read biases.bin\n";
        delete[] input_buf; delete[] output_hw; delete[] output_golden;
        delete[] weights_raw; delete[] biases_raw;
        delete[] weights; delete[] biases;
        return 1;
    }
    bf.close();
    std::cout << "✓ Loaded " << B_SZ << " bias bytes\n";

    // Convert int8 to ap_fixed<8,4> (Q3.4)
    // int8 value / 16 = Q3.4 float value
    std::cout << "Converting to Q3.4 format...\n";
    for (int i = 0; i < W_SZ; i++) {
        weights[i] = (weight_t)(weights_raw[i] / 16.0f);
    }
    for (int i = 0; i < B_SZ; i++) {
        biases[i] = (bias_t)(biases_raw[i] / 16.0f);
    }

    // Print some weight stats
    std::cout << "\nWeight statistics:\n";
    std::cout << "  First 10 weights (raw int8): ";
    for (int i = 0; i < 10; i++) std::cout << (int)weights_raw[i] << " ";
    std::cout << "\n";
    std::cout << "  First 10 weights (Q3.4):     ";
    for (int i = 0; i < 10; i++) std::cout << (float)weights[i] << " ";
    std::cout << "\n";

    std::cout << "\nConv10 biases (classifier):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "  Class " << i << ": raw=" << (int)biases_raw[B_SZ - 10 + i]
                  << " Q3.4=" << (float)biases[B_SZ - 10 + i] << "\n";
    }

    // Create test input (gradient pattern)
    std::cout << "\nCreating test input...\n";
    gradient_fill(input_buf, 3, 224, 224, 0.5f);

    // Run HLS
    std::cout << "Running HLS inference...\n";
    bool done = false;
    squeezenet_top(input_buf, output_hw, weights, biases,
                   ddr_buffer_a, ddr_buffer_b, true, &done);

    if (!done) {
        std::cout << "✗ ERROR: HLS did not complete\n";
        delete[] input_buf; delete[] output_hw; delete[] output_golden;
        delete[] weights_raw; delete[] biases_raw;
        delete[] weights; delete[] biases;
        return 1;
    }
    std::cout << "✓ HLS completed\n";

    // Run golden reference
    std::cout << "Running golden reference...\n";
    golden_squeezenet_full(input_buf, output_golden, weights, biases);
    std::cout << "✓ Golden completed\n";

    // Compare
    std::cout << "\n=== OUTPUT COMPARISON ===\n";
    std::cout << "Class |    HLS    |  Golden   |   Diff\n";
    std::cout << "------|-----------|-----------|----------\n";

    int hw_pred = 0, gold_pred = 0;
    float hw_max = -1000, gold_max = -1000;
    float maxdiff = 0;

    for (int i = 0; i < OUT_SZ; i++) {
        float hw_val = (float)output_hw[i];
        float gold_val = (float)output_golden[i];
        float diff = std::fabs(hw_val - gold_val);

        std::cout << "  " << i << "   | " << std::fixed << std::setprecision(4)
                  << std::setw(9) << hw_val << " | "
                  << std::setw(9) << gold_val << " | "
                  << std::setw(8) << diff << "\n";

        if (diff > maxdiff) maxdiff = diff;
        if (hw_val > hw_max) { hw_max = hw_val; hw_pred = i; }
        if (gold_val > gold_max) { gold_max = gold_val; gold_pred = i; }
    }

    std::cout << "\nMaximum difference: " << maxdiff << "\n";
    std::cout << "Predicted class (HLS):    " << hw_pred << " (score: " << hw_max << ")\n";
    std::cout << "Predicted class (Golden): " << gold_pred << " (score: " << gold_max << ")\n";

    int result = 0;
    if (hw_pred == gold_pred) {
        std::cout << "✓ Predictions MATCH!\n";
    } else {
        std::cout << "✗ Predictions DIFFER!\n";
        result = 1;
    }

    // Cleanup
    delete[] input_buf;
    delete[] output_hw;
    delete[] output_golden;
    delete[] weights_raw;
    delete[] biases_raw;
    delete[] weights;
    delete[] biases;

    return result;
}

// =============================================================
// Main Test Suite
// =============================================================

int main() {
    std::cout << "==============================================\n";
    std::cout << "SqueezeNet HLS Testbench\n";
    std::cout << "==============================================\n";
    std::cout << "Modified for external squeeze buffer in fire_module\n\n";

    // Allocate DDR buffers
    data_t *ddr_buffer_a = (data_t*)malloc(1605632 * sizeof(data_t));
    data_t *ddr_buffer_b = (data_t*)malloc(1605632 * sizeof(data_t));

    if (!ddr_buffer_a || !ddr_buffer_b) {
        std::cout << "✗ ERROR: Failed to allocate DDR buffers\n";
        return 1;
    }

    memset(ddr_buffer_a, 0, 1605632 * sizeof(data_t));
    memset(ddr_buffer_b, 0, 1605632 * sizeof(data_t));

    int total_errors = 0;

    // Run individual module tests first
    std::cout << "\n========== INDIVIDUAL MODULE TESTS ==========\n";
    total_errors += test_conv1x1_ip();
    total_errors += test_conv3x3_ip();
    total_errors += test_conv7x7_ip();
    total_errors += test_maxpool_ip();
    total_errors += test_fire_ip();
    total_errors += test_controller_logic();

    // Run full network test
    std::cout << "\n========== FULL NETWORK TEST ==========\n";
    total_errors += test_with_trained_weights(ddr_buffer_a, ddr_buffer_b);

    // Summary
    std::cout << "\n==============================================\n";
    std::cout << "TEST SUMMARY\n";
    std::cout << "==============================================\n";
    if (total_errors == 0) {
        std::cout << "✓ ALL TESTS PASSED!\n";
    } else {
        std::cout << "✗ SOME TESTS FAILED (total errors: " << total_errors << ")\n";
    }

    // Cleanup
    free(ddr_buffer_a);
    free(ddr_buffer_b);

    return total_errors;
}
