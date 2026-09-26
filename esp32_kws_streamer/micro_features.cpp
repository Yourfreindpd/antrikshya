/**
 * =============================================================================
 * ISRO SIH 26172: ESP32 Micro-MFCC Feature Extractor Implementation
 * =============================================================================
 */

#include "micro_features.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

MicroFeatures::MicroFeatures() {}

static inline float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

bool MicroFeatures::init() {
    // 1. Precalculate Hamming Window
    for (int i = 0; i < MFCC_FRAME_LEN; i++) {
        hamming_window[i] = 0.54f - 0.46f * cosf((2.0f * (float)M_PI * i) / (MFCC_FRAME_LEN - 1));
    }

    // 2. Precalculate Mel Filterbank Frequency Bin Centers
    float mel_min = hz_to_mel(300.0f);
    float mel_max = hz_to_mel(8000.0f);
    float mel_step = (mel_max - mel_min) / (MFCC_NUM_MELS + 1);

    for (int i = 0; i < MFCC_NUM_MELS + 2; i++) {
        float mel = mel_min + i * mel_step;
        float hz = mel_to_hz(mel);
        mel_filter_bins[i] = (int)floorf((MFCC_FFT_SIZE + 1) * hz / 16000.0f);
        if (mel_filter_bins[i] > MFCC_FFT_SIZE / 2) {
            mel_filter_bins[i] = MFCC_FFT_SIZE / 2;
        }
    }

    // 3. Precalculate Ortho DCT-II Basis Matrix
    for (int k = 0; k < MFCC_NUM_COEFFS; k++) {
        float factor = (k == 0) ? sqrtf(1.0f / MFCC_NUM_MELS) : sqrtf(2.0f / MFCC_NUM_MELS);
        for (int m = 0; m < MFCC_NUM_MELS; m++) {
            dct_basis[k * MFCC_NUM_MELS + m] = factor * cosf(((float)M_PI * (m + 0.5f) * k) / MFCC_NUM_MELS);
        }
    }

    // 4. Allocate Scratch Buffers on Heap to protect task stack
    frame_buf = (float*)malloc(MFCC_FRAME_LEN * sizeof(float));
    fft_real = (float*)malloc(MFCC_FFT_SIZE * sizeof(float));
    fft_imag = (float*)malloc(MFCC_FFT_SIZE * sizeof(float));
    power_spec = (float*)malloc((MFCC_FFT_SIZE / 2 + 1) * sizeof(float));
    mel_energies = (float*)malloc(MFCC_NUM_MELS * sizeof(float));

    if (!frame_buf || !fft_real || !fft_imag || !power_spec || !mel_energies) {
        return false;
    }

    return true;
}

MicroFeatures::~MicroFeatures() {
    if (frame_buf) free(frame_buf);
    if (fft_real) free(fft_real);
    if (fft_imag) free(fft_imag);
    if (power_spec) free(power_spec);
    if (mel_energies) free(mel_energies);
}

float MicroFeatures::calculate_rms(const int16_t* samples, int count) {
    if (count <= 0) return 0.0f;
    float sum_sq = 0.0f;
    // Downsample check by 2 for ultra-fast CPU savings
    for (int i = 0; i < count; i += 2) {
        float val = (float)samples[i];
        sum_sq += val * val;
    }
    return sqrtf(sum_sq / (count / 2.0f));
}

// Cooley-Tukey Radix-2 Decimation-in-Time FFT
void MicroFeatures::compute_fft(const float* input, float* real, float* imag, int n) {
    for (int i = 0; i < n; i++) {
        real[i] = (i < MFCC_FRAME_LEN) ? input[i] : 0.0f;
        imag[i] = 0.0f;
        
    }

    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
        int k = n >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    // Radix-2 butterfly computation
    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * (float)M_PI / len;
        float wlen_r = cosf(angle);
        float wlen_i = sinf(angle);
        int half_len = len >> 1;

        for (int i = 0; i < n; i += len) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            for (int k = 0; k < half_len; k++) {
                int u = i + k;
                int v = i + k + half_len;
                float vr = real[v] * w_r - imag[v] * w_i;
                float vi = real[v] * w_i + imag[v] * w_r;

                real[v] = real[u] - vr;
                imag[v] = imag[u] - vi;
                real[u] += vr;
                imag[u] += vi;

                float next_w_r = w_r * wlen_r - w_i * wlen_i;
                float next_w_i = w_r * wlen_i + w_i * wlen_r;
                w_r = next_w_r;
                w_i = next_w_i;
            }
        }
    }
}

bool MicroFeatures::extract_features(const int16_t* input_samples, int num_samples, float* output_features) {
    if (num_samples < 16000 || !output_features || !frame_buf) return false;

    float mean_acc = 0.0f;
    float sum_sq_acc = 0.0f;
    int total_coeffs = MFCC_NUM_FRAMES * MFCC_NUM_COEFFS;

    for (int frame_idx = 0; frame_idx < MFCC_NUM_FRAMES; frame_idx++) {
        int start = frame_idx * MFCC_FRAME_STEP;

        // Apply pre-emphasis and Hamming window
        for (int i = 0; i < MFCC_FRAME_LEN; i++) {
            float current = (float)input_samples[start + i] / 32768.0f;
            float prev = (start + i > 0) ? (float)input_samples[start + i - 1] / 32768.0f : current;
            float emphasized = current - 0.97f * prev;
            frame_buf[i] = emphasized * hamming_window[i];
        }

        // FFT (512-point)
        compute_fft(frame_buf, fft_real, fft_imag, MFCC_FFT_SIZE);

        // Power spectrum (normalized)
        for (int i = 0; i <= MFCC_FFT_SIZE / 2; i++) {
            power_spec[i] = (fft_real[i] * fft_real[i] + fft_imag[i] * fft_imag[i]) / (float)MFCC_FFT_SIZE;
        }

        // Mel Filterbank integration
        for (int m = 0; m < MFCC_NUM_MELS; m++) {
            float energy = 0.0f;
            int left = mel_filter_bins[m];
            int center = mel_filter_bins[m + 1];
            int right = mel_filter_bins[m + 2];

            for (int f = left; f < center; f++) {
                energy += power_spec[f] * ((float)(f - left) / (float)(center - left + 1e-6f));
            }
            for (int f = center; f < right; f++) {
                energy += power_spec[f] * ((float)(right - f) / (float)(right - center + 1e-6f));
            }
            mel_energies[m] = logf(fmaxf(energy, 1e-6f));
        }

        // DCT-II Matrix multiplication
        int out_offset = frame_idx * MFCC_NUM_COEFFS;
        for (int k = 0; k < MFCC_NUM_COEFFS; k++) {
            float ceps = 0.0f;
            int basis_offset = k * MFCC_NUM_MELS;
            for (int m = 0; m < MFCC_NUM_MELS; m++) {
                ceps += mel_energies[m] * dct_basis[basis_offset + m];
            }
            output_features[out_offset + k] = ceps;
            mean_acc += ceps;
        }
    }

    // Clip-level standardization
    float mean = mean_acc / total_coeffs;
    for (int i = 0; i < total_coeffs; i++) {
        float diff = output_features[i] - mean;
        sum_sq_acc += diff * diff;
    }
    float std_dev = sqrtf(sum_sq_acc / total_coeffs) + 1e-6f;

    for (int i = 0; i < total_coeffs; i++) {
        output_features[i] = (output_features[i] - mean) / std_dev;
    }

    return true;
}
