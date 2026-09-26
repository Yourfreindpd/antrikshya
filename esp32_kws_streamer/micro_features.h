/**
 * =============================================================================
 * ISRO SIH 26172: ESP32 Micro-MFCC Feature Extractor
 * =============================================================================
 * Ultra-compact, deterministic MFCC computation designed for ESP32 SRAM:
 * - Computes 49 frames x 13 MFCC coefficients from 16,000 samples (1.0 sec)
 * - Uses minimal scratchpad RAM (< 8 KB)
 * =============================================================================
 */

#ifndef MICRO_FEATURES_H
#define MICRO_FEATURES_H

#include <stdint.h>
#include <stddef.h>

#define MFCC_NUM_FRAMES 49
#define MFCC_NUM_COEFFS 13
#define MFCC_NUM_MELS   40
#define MFCC_FFT_SIZE   512
#define MFCC_FRAME_LEN  400   // 25ms @ 16kHz
#define MFCC_FRAME_STEP 320   // 20ms @ 16kHz

class MicroFeatures {
public:
    MicroFeatures();
    ~MicroFeatures();

    // Initializes Hamming window and Mel filterbank tables
    bool init();

    // Computes (49, 13) MFCC matrix from raw 16-bit PCM audio samples
    // input_samples: 16,000 int16 samples
    // output_features: float array of size [49 * 13]
    bool extract_features(const int16_t* input_samples, int num_samples, float* output_features);

    // Fast RMS calculation for Tier 1 VAD (< 5% CPU)
    static float calculate_rms(const int16_t* samples, int count);

private:
    float hamming_window[MFCC_FRAME_LEN];
    float dct_basis[MFCC_NUM_COEFFS * MFCC_NUM_MELS];
    int mel_filter_bins[MFCC_NUM_MELS + 2];

    float* frame_buf = nullptr;
    float* fft_real = nullptr;
    float* fft_imag = nullptr;
    float* power_spec = nullptr;
    float* mel_energies = nullptr;

    void compute_fft(const float* input, float* real, float* imag, int n);
};

#endif // MICRO_FEATURES_H
