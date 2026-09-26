/**
 * =============================================================================
 * ISRO SIH 26172: ESP32 Voice Activator Configuration
 * =============================================================================
 * Target Keyword: "antriksya"
 * Constraints: < 256KB RAM, < 10% Idle CPU, Low-latency streaming
 * =============================================================================
 */

#ifndef CONFIG_H
#define CONFIG_H

// =============================================================================
// I2S MICROPHONE PINS (Default: INMP441 / SPH0645)
// =============================================================================
#define I2S_SCK_PIN 14 // Serial Clock (BCLK)
#define I2S_WS_PIN 15  // Word Select (LRC / LRCLK)
#define I2S_SD_PIN 32  // Serial Data In (DOUT from mic)
#define I2S_PORT I2S_NUM_0

// Audio format
#define SAMPLE_RATE 16000  // 16 kHz
#define BITS_PER_SAMPLE 16 // 16-bit PCM
#define AUDIO_CHANNELS 1   // Mono

// Buffer dimensions
#define WINDOW_DURATION_MS 1000 // 1.0 second sliding window for KWS
#define SAMPLES_PER_WINDOW                                                     \
  ((SAMPLE_RATE * WINDOW_DURATION_MS) / 1000) // 16,000 samples
#define RING_BUFFER_MS 1500                   // 1.5 seconds circular storage
#define RING_BUFFER_SAMPLES                                                    \
  ((SAMPLE_RATE * RING_BUFFER_MS) / 1000) // 24,000 samples = 48 KB

// =============================================================================
// TIER 1: VAD (VOICE ACTIVITY DETECTION) THRESHOLDS
// Guarantees < 10% CPU during idle / silence listening
// =============================================================================
#define VAD_ENERGY_THRESHOLD                                                   \
  450 // Minimum RMS amplitude to trigger ML inference
#define VAD_CHECK_INTERVAL_MS 30 // Check interval when idle

// =============================================================================
// TIER 2: KEYWORD SPOTTER THRESHOLDS
// =============================================================================
#define KWS_CONFIDENCE_THRESHOLD                                               \
  0.82f // Required softmax probability for "antriksya"
#define MFCC_FRAMES 49
#define MFCC_COEFFS 13

// =============================================================================
// WI-FI & ASR WEBSOCKET CONFIGURATION
// =============================================================================
#define WIFI_SSID "praneet"
#define WIFI_PASSWORD "12345678"

// Remote ASR Server (IP address of your PC running asr_server.py)
#define ASR_SERVER_HOST "10.135.131.188"
#define ASR_SERVER_PORT 8000
#define ASR_WEBSOCKET_PATH "/ws/audio"

// Streaming Parameters
#define STREAM_CHUNK_MS 100 // Stream audio in 100ms packets to minimize latency
#define STREAM_CHUNK_SAMPLES                                                   \
  ((SAMPLE_RATE * STREAM_CHUNK_MS) / 1000) // 1,600 samples
#define STREAM_PRE_ROLL_MS                                                     \
  250 // Send 250ms of audio before trigger so command isn't clipped
#define STREAM_MAX_DURATION_SEC 6 // Max command streaming length before timeout
#define STREAM_SILENCE_TIMEOUT_MS                                              \
  1200 // Stop streaming if user stops talking for 1.2 seconds

#endif // CONFIG_H
