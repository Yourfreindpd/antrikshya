#include "micro_features.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: mfcc_host_runner <input.raw> <output.csv>\n";
        return 2;
    }

    std::ifstream input(argv[1], std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open input: " << argv[1] << "\n";
        return 1;
    }

    std::vector<int16_t> samples(16000);
    input.read(reinterpret_cast<char*>(samples.data()),
               static_cast<std::streamsize>(samples.size() * sizeof(int16_t)));
    if (input.gcount() != static_cast<std::streamsize>(samples.size() * sizeof(int16_t))) {
        std::cerr << "Input must contain exactly 16000 int16 samples\n";
        return 1;
    }

    MicroFeatures features;
    if (!features.init()) {
        std::cerr << "MFCC initialization failed\n";
        return 1;
    }

    float output[MFCC_NUM_FRAMES * MFCC_NUM_COEFFS];
    if (!features.extract_features(samples.data(), static_cast<int>(samples.size()), output)) {
        std::cerr << "MFCC extraction failed\n";
        return 1;
    }

    std::ofstream csv(argv[2]);
    if (!csv) {
        std::cerr << "Cannot open output: " << argv[2] << "\n";
        return 1;
    }

    csv.setf(std::ios::scientific);
    csv.precision(9);
    for (int frame = 0; frame < MFCC_NUM_FRAMES; ++frame) {
        for (int coefficient = 0; coefficient < MFCC_NUM_COEFFS; ++coefficient) {
            if (coefficient > 0) csv << ',';
            csv << output[frame * MFCC_NUM_COEFFS + coefficient];
        }
        csv << '\n';
    }
}