"""Compare the ESP32 MFCC implementation with an independent Python port.

This test uses generated int16 PCM, so it does not require an ESP32 or microphone.
The C++ result comes from the production micro_features.cpp implementation.
"""

from pathlib import Path
import subprocess
import sys

import numpy as np


SAMPLE_RATE = 16000
NUM_SAMPLES = 16000
FRAME_LENGTH = 400
FRAME_STEP = 320
FFT_SIZE = 512
NUM_FRAMES = 49
NUM_MELS = 40
NUM_COEFFS = 13

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
BUILD = TOOLS / "mfcc_host_runner.exe"
INPUT = TOOLS / "mfcc_test_input.raw"
CPP_OUTPUT = TOOLS / "mfcc_cpp_output.csv"


def create_test_audio() -> np.ndarray:
    """Create reproducible PCM containing tones, speech-like modulation, and noise."""
    time = np.arange(NUM_SAMPLES, dtype=np.float64) / SAMPLE_RATE
    signal = (
        0.35 * np.sin(2 * np.pi * 440 * time)
        + 0.20 * np.sin(2 * np.pi * 1200 * time)
        + 0.10 * np.sin(2 * np.pi * 2300 * time)
    )
    modulation = 0.65 + 0.35 * np.sin(2 * np.pi * 3 * time)
    noise = 0.015 * np.sin(2 * np.pi * 71 * time)
    return np.rint(np.clip(signal * modulation + noise, -1, 1) * 32767).astype(np.int16)


def hz_to_mel(hz: float) -> np.float32:
    return np.float32(2595.0 * np.log10(1.0 + hz / 700.0))


def mel_to_hz(mel: float) -> np.float32:
    return np.float32(700.0 * (10.0 ** (mel / 2595.0) - 1.0))


def esp32_fft(input_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the single-precision radix-2 FFT in micro_features.cpp."""
    real = np.zeros(FFT_SIZE, dtype=np.float32)
    imag = np.zeros(FFT_SIZE, dtype=np.float32)
    real[:FRAME_LENGTH] = input_frame

    j = 0
    for index in range(FFT_SIZE - 1):
        if index < j:
            real[index], real[j] = real[j], real[index]
            imag[index], imag[j] = imag[j], imag[index]
        step = FFT_SIZE >> 1
        while step <= j:
            j -= step
            step >>= 1
        j += step

    length = 2
    while length <= FFT_SIZE:
        angle = np.float32(-2.0 * np.pi / length)
        wlen_real = np.float32(np.cos(angle))
        wlen_imag = np.float32(np.sin(angle))
        half_length = length >> 1
        for start in range(0, FFT_SIZE, length):
            w_real = np.float32(1.0)
            w_imag = np.float32(0.0)
            for offset in range(half_length):
                upper = start + offset
                lower = upper + half_length
                value_real = np.float32(real[lower] * w_real - imag[lower] * w_imag)
                value_imag = np.float32(real[lower] * w_imag + imag[lower] * w_real)
                upper_real = real[upper]
                upper_imag = imag[upper]
                real[lower] = np.float32(upper_real - value_real)
                imag[lower] = np.float32(upper_imag - value_imag)
                real[upper] = np.float32(upper_real + value_real)
                imag[upper] = np.float32(upper_imag + value_imag)
                next_real = np.float32(w_real * wlen_real - w_imag * wlen_imag)
                w_imag = np.float32(w_real * wlen_imag + w_imag * wlen_real)
                w_real = next_real
        length <<= 1

    return real, imag


def python_mfcc(samples: np.ndarray) -> np.ndarray:
    hamming = (0.54 - 0.46 * np.cos(
        (2.0 * np.pi * np.arange(FRAME_LENGTH)) / (FRAME_LENGTH - 1)
    )).astype(np.float32)

    mel_min = hz_to_mel(np.float32(300.0))
    mel_max = hz_to_mel(np.float32(8000.0))
    mel_step = np.float32((mel_max - mel_min) / np.float32(NUM_MELS + 1))
    mel_filter_bins = np.empty(NUM_MELS + 2, dtype=int)
    for index in range(NUM_MELS + 2):
        mel = np.float32(mel_min + np.float32(index) * mel_step)
        hz = mel_to_hz(mel)
        value = np.floor(
            np.float32((FFT_SIZE + 1) * hz / np.float32(SAMPLE_RATE))
        )
        mel_filter_bins[index] = min(int(value), FFT_SIZE // 2)

    dct_basis = np.empty((NUM_COEFFS, NUM_MELS), dtype=np.float32)
    for coefficient in range(NUM_COEFFS):
        factor = np.float32(np.sqrt(1.0 / NUM_MELS if coefficient == 0 else 2.0 / NUM_MELS))
        dct_basis[coefficient] = factor * np.cos(
            np.pi * (np.arange(NUM_MELS) + 0.5) * coefficient / NUM_MELS
        )

    output = np.empty((NUM_FRAMES, NUM_COEFFS), dtype=np.float32)
    scaled = samples.astype(np.float32) / np.float32(32768.0)

    for frame_index in range(NUM_FRAMES):
        start = frame_index * FRAME_STEP
        frame = scaled[start:start + FRAME_LENGTH]
        previous = np.empty(FRAME_LENGTH, dtype=np.float32)
        previous[0] = frame[0]
        previous[1:] = scaled[start:start + FRAME_LENGTH - 1]
        frame = np.asarray((frame - np.float32(0.97) * previous) * hamming, dtype=np.float32)

        fft_real, fft_imag = esp32_fft(frame)
        power = np.asarray(
            (fft_real * fft_real + fft_imag * fft_imag) / np.float32(FFT_SIZE),
            dtype=np.float32,
        )

        mel_energies = np.empty(NUM_MELS, dtype=np.float32)
        for mel_index in range(NUM_MELS):
            left, center, right = mel_filter_bins[mel_index:mel_index + 3]
            energy = np.float32(0.0)
            for frequency in range(left, center):
                energy = np.float32(
                    energy + power[frequency] * np.float32(
                        (frequency - left) / (center - left + 1e-6)
                    )
                )
            for frequency in range(center, right):
                energy = np.float32(
                    energy + power[frequency] * np.float32(
                        (right - frequency) / (right - center + 1e-6)
                    )
                )
            mel_energies[mel_index] = np.float32(np.log(max(float(energy), 1e-6)))

        for coefficient in range(NUM_COEFFS):
            value = np.float32(0.0)
            for mel_index in range(NUM_MELS):
                value = np.float32(
                    value + mel_energies[mel_index] * dct_basis[coefficient, mel_index]
                )
            output[frame_index, coefficient] = value

    mean = np.float32(np.sum(output, dtype=np.float32) / np.float32(NUM_FRAMES * NUM_COEFFS))
    sum_squared = np.float32(0.0)
    for value in output.ravel():
        difference = np.float32(value - mean)
        sum_squared = np.float32(sum_squared + difference * difference)
    standard_deviation = np.float32(
        np.sqrt(sum_squared / np.float32(NUM_FRAMES * NUM_COEFFS)) + np.float32(1e-6)
    )
    for index in np.ndindex(output.shape):
        output[index] = np.float32((output[index] - mean) / standard_deviation)
    return output


def build_runner() -> None:
    compiler = "g++"
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-I",
        str(ROOT / "include"),
        str(TOOLS / "mfcc_host_runner.cpp"),
        str(ROOT / "src" / "micro_features.cpp"),
        "-o",
        str(BUILD),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        raise SystemExit("g++ was not found. Install MinGW or use the PlatformIO toolchain.")


def compare_serial_dump(serial_log: Path) -> int:
    lines = serial_log.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        pcm_start = lines.index("PCM_BEGIN") + 1
        pcm_end = lines.index("PCM_END")
        mfcc_start = lines.index("MFCC_BEGIN") + 1
        mfcc_end = lines.index("MFCC_END")
    except ValueError as error:
        raise SystemExit(f"Serial markers not found: {error}")

    samples = np.asarray([int(value) for value in lines[pcm_start:pcm_end]], dtype=np.int16)
    esp_output = np.loadtxt(lines[mfcc_start:mfcc_end], delimiter=",", dtype=np.float32)
    if samples.size != NUM_SAMPLES:
        raise SystemExit(f"Expected {NUM_SAMPLES} PCM samples, found {samples.size}")

    python_output = python_mfcc(samples)
    difference = esp_output - python_output
    maximum_error = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    correlation = float(np.corrcoef(esp_output.ravel(), python_output.ravel())[0, 1])
    print(f"ESP32 shape:     {esp_output.shape}")
    print(f"Python shape:    {python_output.shape}")
    print(f"Maximum error:   {maximum_error:.8g}")
    print(f"RMSE:            {rmse:.8g}")
    print(f"Correlation:     {correlation:.8f}")
    passed = esp_output.shape == (NUM_FRAMES, NUM_COEFFS) and maximum_error < 0.1 and correlation > 0.999
    print("RESULT: PASS" if passed else "RESULT: FAIL")
    return 0 if passed else 1


def main() -> int:
    if len(sys.argv) == 2:
        return compare_serial_dump(Path(sys.argv[1]))

    build_runner()
    samples = create_test_audio()
    samples.tofile(INPUT)
    subprocess.run([str(BUILD), str(INPUT), str(CPP_OUTPUT)], check=True)

    cpp_output = np.loadtxt(CPP_OUTPUT, delimiter=",", dtype=np.float32)
    python_output = python_mfcc(samples)
    difference = cpp_output - python_output
    maximum_error = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    correlation = float(np.corrcoef(cpp_output.ravel(), python_output.ravel())[0, 1])

    print(f"C++ shape:       {cpp_output.shape}")
    print(f"Python shape:    {python_output.shape}")
    print(f"Maximum error:   {maximum_error:.8g}")
    print(f"RMSE:            {rmse:.8g}")
    print(f"Correlation:     {correlation:.8f}")

    # Different libm implementations (ESP32 cosf/logf versus NumPy) are not
    # bit-identical, so use a tight feature-level tolerance and correlation.
    passed = (
        cpp_output.shape == (NUM_FRAMES, NUM_COEFFS)
        and maximum_error < 0.1
        and correlation > 0.999
    )
    print("RESULT: PASS" if passed else "RESULT: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())