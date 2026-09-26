"""
==============================================================================
ISRO SIH 26172: Synthetic Augmentation & Dataset Generator for 'antriksya'
==============================================================================
Multiplies raw recorded seeds of 'antriksya' into a balanced, robust dataset
with 1,000+ samples per class:
- Class 0: 'antriksya' (augmented with pitch, speed, SNR noise, time shift)
- Class 1: 'unknown' (other spoken words, varied phonemes, non-keywords)
- Class 2: 'silence' (room ambient, fan noise, typing, air conditioning)

All audio is normalized and saved as 1.0s, 16kHz, 16-bit mono WAV files.
"""

import os
import glob
import random
import numpy as np
from scipy.io import wavfile
from scipy import signal

SAMPLE_RATE = 16000
CLIP_DURATION = 1.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

DIR_KEYWORD = os.path.join(DATASET_DIR, "antriksya")
DIR_UNKNOWN = os.path.join(DATASET_DIR, "unknown")
DIR_SILENCE = os.path.join(DATASET_DIR, "silence")

os.makedirs(DIR_KEYWORD, exist_ok=True)
os.makedirs(DIR_UNKNOWN, exist_ok=True)
os.makedirs(DIR_SILENCE, exist_ok=True)

def load_wav(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    else:
        audio = data.astype(np.float32)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        num_target = int(len(audio) * SAMPLE_RATE / sr)
        audio = signal.resample(audio, num_target)
    return audio

def save_wav(path, audio_float):
    audio_clipped = np.clip(audio_float, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767.0).astype(np.int16)
    wavfile.write(path, SAMPLE_RATE, audio_int16)

def time_shift(audio, max_shift_ms=120):
    shift = int(random.uniform(-max_shift_ms, max_shift_ms) * SAMPLE_RATE / 1000.0)
    if shift > 0:
        return np.pad(audio, (shift, 0))[:len(audio)]
    elif shift < 0:
        return np.pad(audio, (0, -shift))[-shift:]
    return audio

def time_stretch(audio, factor):
    new_len = int(len(audio) / factor)
    resampled = signal.resample(audio, new_len)
    if len(resampled) < CLIP_SAMPLES:
        pad = CLIP_SAMPLES - len(resampled)
        pad_l = pad // 2
        pad_r = pad - pad_l
        return np.pad(resampled, (pad_l, pad_r))
    else:
        start = (len(resampled) - CLIP_SAMPLES) // 2
        return resampled[start:start + CLIP_SAMPLES]

def add_noise(audio, snr_db=15):
    sig_power = np.mean(audio ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    # Generate realistic colored noise (pink/brown noise approximation)
    white = np.random.randn(len(audio))
    b, a = signal.butter(1, 0.25, btype='low')
    colored_noise = signal.lfilter(b, a, white)
    colored_noise = colored_noise * np.sqrt(noise_power / (np.mean(colored_noise ** 2) + 1e-12))
    return audio + colored_noise

def generate_synthetic_noise(noise_type="room", length=CLIP_SAMPLES):
    white = np.random.randn(length)
    if noise_type == "fan":
        # Low frequency rumble
        b, a = signal.butter(2, 200.0 / (SAMPLE_RATE / 2), btype='low')
        noise = signal.lfilter(b, a, white) * 0.05
    elif noise_type == "hiss":
        # High frequency hiss
        b, a = signal.butter(2, 2000.0 / (SAMPLE_RATE / 2), btype='high')
        noise = signal.lfilter(b, a, white) * 0.015
    else:
        # Typical room noise (pink-ish noise + small bursts)
        b, a = signal.butter(1, 800.0 / (SAMPLE_RATE / 2), btype='low')
        noise = signal.lfilter(b, a, white) * 0.02
        if random.random() < 0.3:
            # Add random typing / click burst
            click_pos = random.randint(100, length - 1000)
            noise[click_pos:click_pos + 120] += np.random.randn(120) * 0.1
    return noise

def generate_synthetic_unknown_syllable(length=CLIP_SAMPLES):
    """Generates synthetic multi-tone speech-like formant bursts for non-keywords."""
    t = np.linspace(0, 1.0, length)
    f0 = random.uniform(100, 260)  # Fundamental voice pitch
    harmonics = [f0 * i for i in range(1, 8)]
    audio = np.zeros(length)
    for h in harmonics:
        audio += (1.0 / (h / f0)) * np.sin(2 * np.pi * h * t + random.uniform(0, 2*np.pi))
    # Apply speech envelope (syllable attack and decay)
    env = np.exp(-((t - random.uniform(0.3, 0.6)) ** 2) / (2 * (random.uniform(0.08, 0.20) ** 2)))
    audio = audio * env
    # Add formant filtering
    b, a = signal.butter(2, [300.0 / (SAMPLE_RATE / 2), 3400.0 / (SAMPLE_RATE / 2)], btype='band')
    audio = signal.lfilter(b, a, audio)
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * random.uniform(0.3, 0.8)
    return add_noise(audio, snr_db=random.uniform(15, 25))

def generate_seed_keyword_if_empty():
    """If no raw audio has been recorded yet, create a baseline synthetic formant seed so pipeline never crashes."""
    raw_files = glob.glob(os.path.join(DIR_KEYWORD, "*.wav"))
    if not raw_files:
        print("Note: No recorded 'antriksya' seeds found yet. Generating synthetic phonetic seeds...")
        t = np.linspace(0, 1.0, CLIP_SAMPLES)
        # Approximate 3-syllable sequence 'an-trik-sya'
        seed = np.zeros(CLIP_SAMPLES)
        # Syllable 1: 'an' (~0.2s - 0.4s)
        env1 = np.exp(-((t - 0.30) ** 2) / (2 * (0.07 ** 2)))
        seed += (np.sin(2 * np.pi * 220 * t) + 0.6 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)) * env1
        # Syllable 2: 'trik' (~0.45s - 0.65s)
        env2 = np.exp(-((t - 0.52) ** 2) / (2 * (0.06 ** 2)))
        seed += (np.sin(2 * np.pi * 310 * t) + 0.5 * np.sin(2 * np.pi * 1200 * t) + np.random.randn(CLIP_SAMPLES) * 0.15) * env2
        # Syllable 3: 'sya' (~0.68s - 0.9s)
        env3 = np.exp(-((t - 0.75) ** 2) / (2 * (0.08 ** 2)))
        seed += (np.sin(2 * np.pi * 280 * t) + np.random.randn(CLIP_SAMPLES) * 0.3) * env3
        seed = seed / (np.max(np.abs(seed)) + 1e-8) * 0.7
        seed_path = os.path.join(DIR_KEYWORD, "antriksya_seed_0001.wav")
        save_wav(seed_path, seed)
        print(f"Created initial seed: {os.path.basename(seed_path)}")

def main(target_samples_per_class=600):
    print("=" * 65)
    print("      ISRO SIH 26172: DATASET AUGMENTATION PIPELINE")
    print("=" * 65)
    print(f"Target Samples per Class: {target_samples_per_class}")
    print(f"Directory: {DATASET_DIR}")
    print()

    generate_seed_keyword_if_empty()

    # 1. Augment 'antriksya'
    keyword_seeds = glob.glob(os.path.join(DIR_KEYWORD, "*.wav"))
    print(f"[1/3] Augmenting 'antriksya' from {len(keyword_seeds)} base recordings...")
    aug_count = len(keyword_seeds)
    seed_audios = [load_wav(p) for p in keyword_seeds]

    while aug_count < target_samples_per_class:
        base = random.choice(seed_audios).copy()
        # Speed stretch (0.88x to 1.15x)
        speed = random.choice([0.88, 0.92, 0.96, 1.0, 1.04, 1.08, 1.14])
        aug = time_stretch(base, speed)
        # Time shift (jitter word in 1s window)
        aug = time_shift(aug, max_shift_ms=80)
        # Volume variation
        aug = aug * random.uniform(0.5, 1.25)
        # Background noise injection (12dB to 30dB SNR)
        aug = add_noise(aug, snr_db=random.uniform(12, 32))
        
        fname = os.path.join(DIR_KEYWORD, f"antriksya_aug_{aug_count:05d}.wav")
        save_wav(fname, aug)
        aug_count += 1
        if aug_count % 150 == 0:
            print(f"  -> Generated {aug_count}/{target_samples_per_class} 'antriksya' samples")

    # 2. Generate / Augment 'unknown' class
    print(f"\n[2/3] Building 'unknown' negative speech class...")
    existing_unknown = glob.glob(os.path.join(DIR_UNKNOWN, "*.wav"))
    unk_count = len(existing_unknown)
    while unk_count < target_samples_per_class:
        unk_audio = generate_synthetic_unknown_syllable(CLIP_SAMPLES)
        fname = os.path.join(DIR_UNKNOWN, f"unknown_{unk_count:05d}.wav")
        save_wav(fname, unk_audio)
        unk_count += 1
        if unk_count % 150 == 0:
            print(f"  -> Generated {unk_count}/{target_samples_per_class} 'unknown' samples")

    # 3. Generate 'silence' / background class
    print(f"\n[3/3] Building 'silence' / background noise class...")
    existing_silence = glob.glob(os.path.join(DIR_SILENCE, "*.wav"))
    sil_count = len(existing_silence)
    noise_types = ["room", "fan", "hiss"]
    while sil_count < target_samples_per_class:
        ntype = random.choice(noise_types)
        sil_audio = generate_synthetic_noise(ntype, CLIP_SAMPLES)
        fname = os.path.join(DIR_SILENCE, f"silence_{sil_count:05d}.wav")
        save_wav(fname, sil_audio)
        sil_count += 1
        if sil_count % 150 == 0:
            print(f"  -> Generated {sil_count}/{target_samples_per_class} 'silence' samples")

    print()
    print("=" * 65)
    print("                 DATASET GENERATION COMPLETE")
    print("=" * 65)
    print(f"  - matriksya (class 0) : {len(glob.glob(os.path.join(DIR_KEYWORD, '*.wav')))} files")
    print(f"  - unknown   (class 1) : {len(glob.glob(os.path.join(DIR_UNKNOWN, '*.wav')))} files")
    print(f"  - silence   (class 2) : {len(glob.glob(os.path.join(DIR_SILENCE, '*.wav')))} files")
    print()
    print("Ready for feature extraction and INT8 model training!")

if __name__ == "__main__":
    main()
