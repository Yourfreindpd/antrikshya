"""Compare one WAV file with the ESP32 MicroFeatures implementation.

The Python implementation intentionally follows src/micro_features.cpp rather
than using librosa, so the result can be compared with the firmware output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

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
CPP_RUNNER = ROOT / "tools" / "mfcc_host_runner.exe"


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE:
            raise ValueError(f"WAV must be {SAMPLE_RATE} Hz, got {wav.getframerate()} Hz")
        if wav.getnchannels() != 1:
            raise ValueError(f"WAV must be mono, got {wav.getnchannels()} channels")
        if wav.getsampwidth() != 2:
            raise ValueError("WAV must contain 16-bit PCM samples")
        raw = wav.readframes(wav.getnframes())

    samples = np.frombuffer(raw, dtype="<i2")
    if len(samples) == 0:
        raise ValueError("WAV contains no samples")
    if len(samples) < NUM_SAMPLES:
        samples = np.pad(samples, (0, NUM_SAMPLES - len(samples)))
    return samples[:NUM_SAMPLES].astype(np.int16, copy=False)


def esp32_fft(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    real = np.zeros(FFT_SIZE, dtype=np.float32)
    imag = np.zeros(FFT_SIZE, dtype=np.float32)
    real[:FRAME_LENGTH] = frame

    bit_reversed = 0
    for index in range(FFT_SIZE - 1):
        if index < bit_reversed:
            real[index], real[bit_reversed] = real[bit_reversed], real[index]
            imag[index], imag[bit_reversed] = imag[bit_reversed], imag[index]
        step = FFT_SIZE >> 1
        while step <= bit_reversed:
            bit_reversed -= step
            step >>= 1
        bit_reversed += step

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
        2.0 * np.pi * np.arange(FRAME_LENGTH) / (FRAME_LENGTH - 1)
    )).astype(np.float32)

    hz_to_mel = lambda hz: np.float32(2595.0 * np.log10(1.0 + hz / 700.0))
    mel_to_hz = lambda mel: np.float32(700.0 * (10.0 ** (mel / 2595.0) - 1.0))
    mel_min = hz_to_mel(np.float32(300.0))
    mel_max = hz_to_mel(np.float32(8000.0))
    mel_step = np.float32((mel_max - mel_min) / np.float32(NUM_MELS + 1))
    mel_bins = np.empty(NUM_MELS + 2, dtype=int)
    for index in range(NUM_MELS + 2):
        mel = np.float32(mel_min + np.float32(index) * mel_step)
        mel_bins[index] = min(
            int(np.floor(np.float32((FFT_SIZE + 1) * mel_to_hz(mel) / SAMPLE_RATE))),
            FFT_SIZE // 2,
        )

    dct_basis = np.empty((NUM_COEFFS, NUM_MELS), dtype=np.float32)
    for coefficient in range(NUM_COEFFS):
        factor = np.float32(np.sqrt(1.0 / NUM_MELS if coefficient == 0 else 2.0 / NUM_MELS))
        dct_basis[coefficient] = factor * np.cos(
            np.pi * (np.arange(NUM_MELS) + 0.5) * coefficient / NUM_MELS
        )

    scaled = samples.astype(np.float32) / np.float32(32768.0)
    output = np.empty((NUM_FRAMES, NUM_COEFFS), dtype=np.float32)
    for frame_index in range(NUM_FRAMES):
        start = frame_index * FRAME_STEP
        frame = scaled[start:start + FRAME_LENGTH]
        previous = np.empty(FRAME_LENGTH, dtype=np.float32)
        previous[0] = frame[0]
        previous[1:] = scaled[start:start + FRAME_LENGTH - 1]
        frame = np.asarray((frame - np.float32(0.97) * previous) * hamming, dtype=np.float32)
        real, imag = esp32_fft(frame)
        power = np.asarray((real * real + imag * imag) / FFT_SIZE, dtype=np.float32)

        mel_energies = np.empty(NUM_MELS, dtype=np.float32)
        for mel_index in range(NUM_MELS):
            left, center, right = mel_bins[mel_index:mel_index + 3]
            energy = np.float32(0.0)
            for frequency in range(left, center):
                energy = np.float32(energy + power[frequency] * ((frequency - left) / (center - left + 1e-6)))
            for frequency in range(center, right):
                energy = np.float32(energy + power[frequency] * ((right - frequency) / (right - center + 1e-6)))
            mel_energies[mel_index] = np.float32(np.log(max(float(energy), 1e-6)))

        for coefficient in range(NUM_COEFFS):
            value = np.float32(0.0)
            for mel_index in range(NUM_MELS):
                value = np.float32(value + mel_energies[mel_index] * dct_basis[coefficient, mel_index])
            output[frame_index, coefficient] = value

    mean = np.float32(np.sum(output, dtype=np.float32) / (NUM_FRAMES * NUM_COEFFS))
    variance = np.float32(0.0)
    for value in output.ravel():
        difference = np.float32(value - mean)
        variance = np.float32(variance + difference * difference)
    standard_deviation = np.float32(np.sqrt(variance / (NUM_FRAMES * NUM_COEFFS)) + 1e-6)
    return np.asarray((output - mean) / standard_deviation, dtype=np.float32)


def build_runner() -> None:
    command = [
        "g++", "-std=c++17", "-O2", "-I", str(ROOT / "include"),
        str(ROOT / "tools" / "mfcc_host_runner.cpp"),
        str(ROOT / "src" / "micro_features.cpp"), "-o", str(CPP_RUNNER),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise SystemExit("g++ is required to run the exact firmware reference") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, help="16 kHz, 16-bit PCM mono WAV")
    parser.add_argument("--output", type=Path, help="write Python MFCC values to CSV")
    args = parser.parse_args()

    samples = read_wav(args.wav)
    python_output = python_mfcc(samples)
    if args.output:
        np.savetxt(args.output, python_output, delimiter=",", fmt="%.9e")

    build_runner()
    with tempfile.TemporaryDirectory() as temporary_directory:
        raw_path = Path(temporary_directory) / "input.raw"
        cpp_path = Path(temporary_directory) / "cpp.csv"
        samples.tofile(raw_path)
        subprocess.run([str(CPP_RUNNER), str(raw_path), str(cpp_path)], check=True)
        cpp_output = np.loadtxt(cpp_path, delimiter=",", dtype=np.float32)

    difference = cpp_output - python_output
    maximum_error = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    correlation = float(np.corrcoef(cpp_output.ravel(), python_output.ravel())[0, 1])
    print(f"C++/ESP32 shape: {cpp_output.shape}")
    print(f"Python shape:    {python_output.shape}")
    print(f"Maximum error:   {maximum_error:.8g}")
    print(f"RMSE:            {rmse:.8g}")
    print(f"Correlation:     {correlation:.8f}")
    passed = maximum_error < 0.1 and correlation > 0.999
    print("RESULT: PASS" if passed else "RESULT: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())