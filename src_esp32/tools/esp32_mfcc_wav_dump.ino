/*
 * Standalone ESP32 MFCC dump sketch.
 *
 * Upload a 16 kHz, 16-bit PCM mono file as /input.wav to LittleFS, then open
 * Serial at 115200. The sketch prints MFCC_BEGIN, 49 CSV rows, and MFCC_END.
 * It uses the production MicroFeatures implementation from this project.
 *
 * For PlatformIO, copy this file to src/main.cpp temporarily (or use it as
 * the only source in a separate project) and make micro_features.cpp available
 * with the project's include/ and src/ directories.
 */

#include <Arduino.h>
#include <LittleFS.h>

#include "micro_features.h"

static constexpr size_t WAV_HEADER_SIZE = 44;
static constexpr size_t SAMPLE_COUNT = 16000;
static int16_t samples[SAMPLE_COUNT];

bool read_input_wav() {
    File wav = LittleFS.open("/input.wav", "r");
    if (!wav || wav.size() < WAV_HEADER_SIZE) {
        Serial.println("ERROR: /input.wav not found or too short");
        return false;
    }

    uint8_t header[WAV_HEADER_SIZE];
    if (wav.read(header, sizeof(header)) != sizeof(header) ||
        memcmp(header, "RIFF", 4) != 0 || memcmp(header + 8, "WAVE", 4) != 0) {
        Serial.println("ERROR: expected a RIFF/WAVE file");
        return false;
    }
    uint16_t format = header[20] | (header[21] << 8);
    uint16_t channels = header[22] | (header[23] << 8);
    uint32_t sample_rate = header[24] | (header[25] << 8) |
                           (header[26] << 16) | (header[27] << 24);
    uint16_t bits = header[34] | (header[35] << 8);
    if (format != 1 || channels != 1 || sample_rate != 16000 || bits != 16) {
        Serial.println("ERROR: WAV must be mono 16 kHz 16-bit PCM");
        return false;
    }

    memset(samples, 0, sizeof(samples));
    size_t bytes_read = wav.read(reinterpret_cast<uint8_t*>(samples), sizeof(samples));
    if (bytes_read < sizeof(samples)) {
        Serial.printf("INFO: padded %u missing samples with silence\n",
                      static_cast<unsigned>((sizeof(samples) - bytes_read) / sizeof(int16_t)));
    }
    return true;
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    if (!LittleFS.begin(true) || !read_input_wav()) {
        return;
    }

    MicroFeatures features;
    float output[MFCC_NUM_FRAMES * MFCC_NUM_COEFFS];
    if (!features.init() || !features.extract_features(samples, SAMPLE_COUNT, output)) {
        Serial.println("ERROR: MFCC extraction failed");
        return;
    }

    Serial.println("MFCC_BEGIN");
    for (int frame = 0; frame < MFCC_NUM_FRAMES; ++frame) {
        for (int coefficient = 0; coefficient < MFCC_NUM_COEFFS; ++coefficient) {
            if (coefficient > 0) Serial.print(',');
            Serial.printf("%.9e", output[frame * MFCC_NUM_COEFFS + coefficient]);
        }
        Serial.println();
    }
    Serial.println("MFCC_END");
}

void loop() {
    delay(1000);
}