"""
==============================================================================
ISRO SIH 26172: Custom Keyword Data Collector for 'antriksya'
==============================================================================
Records high-quality 16kHz 16-bit mono training audio for the custom keyword
'antriksya' (and its phonetic variations like 'antriksh').

Features:
- Fixed 16 kHz 16-bit mono format (microcontroller standard)
- Energy-based automatic endpointing and word centering (1.0 second clip)
- Interactive guided takes (normal, whisper, fast, slow, distant, pitch variations)
- Saves audio directly to dataset/antriksya/
"""

import os
import sys
import time
import numpy as np

try:
    import sounddevice as sd
    from scipy.io import wavfile
except ImportError:
    print("Error: Missing sounddevice or scipy. Run: pip install sounddevice scipy")
    sys.exit(1)

SAMPLE_RATE = 16000
CHANNELS = 1
CLIP_DURATION = 1.0  # 1.0 second window for the KWS model
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)
RECORD_WINDOW = 2.0  # Record 2.0s to allow natural speech timing, then auto-center

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dataset", "antriksya")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TAKE_PROMPTS = [
    "Normal clear voice: 'Antriksya'",
    "Normal voice (slight variation): 'Antriksh'",
    "Slightly faster: 'Antriksya'",
    "Slightly slower / drawn out: 'An-trik-sya'",
    "Whispered / quiet voice: 'antriksya'",
    "Authoritative / louder voice: 'ANTRIKSYA'",
    "Further from mic (step back 1-2 feet): 'Antriksya'",
    "Higher pitch: 'Antriksya'",
    "Lower / deeper pitch: 'Antriksya'",
    "Normal tone with background fan / ambient sound: 'Antriksya'"
]

def find_word_boundaries(audio_1d, threshold=600, min_len_ms=180):
    """Detect start and end of spoken word based on signal amplitude."""
    min_samples = int((min_len_ms / 1000.0) * SAMPLE_RATE)
    above = np.abs(audio_1d) > threshold
    if not np.any(above):
        return None

    # Find contiguous speech segments
    diff = np.diff(above.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        ends = np.append(ends, len(above))

    runs = [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_samples]
    if not runs:
        # Fallback to widest active region
        active = np.where(above)[0]
        return (active[0], active[-1])

    # Pick the longest contiguous region
    return max(runs, key=lambda r: r[1] - r[0])

def center_clip(audio_1d, start_idx, end_idx):
    """Centers the detected word inside a strict 1-second (16000 samples) window."""
    center = (start_idx + end_idx) // 2
    half_window = CLIP_SAMPLES // 2

    c_start = center - half_window
    c_end = c_start + CLIP_SAMPLES

    if c_start < 0:
        c_end -= c_start
        c_start = 0
    if c_end > len(audio_1d):
        shift = c_end - len(audio_1d)
        c_start = max(0, c_start - shift)
        c_end = len(audio_1d)

    clip = audio_1d[c_start:c_end]
    if len(clip) < CLIP_SAMPLES:
        pad_left = (CLIP_SAMPLES - len(clip)) // 2
        pad_right = CLIP_SAMPLES - len(clip) - pad_left
        clip = np.pad(clip, (pad_left, pad_right))

    return clip

def get_next_filename(folder):
    existing = [f for f in os.listdir(folder) if f.startswith("antriksya_") and f.endswith(".wav")]
    numbers = []
    for f in existing:
        try:
            num = int(f.split("_")[1].split(".")[0])
            numbers.append(num)
        except Exception:
            pass
    next_num = (max(numbers) + 1) if numbers else 1
    return os.path.join(folder, f"antriksya_{next_num:04d}.wav"), next_num

def main():
    print("=" * 65)
    print("   ISRO SIH 26172 - KEYWORD DATA COLLECTOR: 'antriksya'")
    print("=" * 65)
    print(f"Target Directory: {OUTPUT_DIR}")
    print(f"Format: {SAMPLE_RATE} Hz, Mono, 16-bit PCM (1.0 second clips)")
    print()

    try:
        dev = sd.query_devices(kind='input')
        print(f"Microphone detected: {dev['name']}")
    except Exception as e:
        print(f"Warning: Could not query default input device: {e}")

    existing_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")])
    print(f"Existing 'antriksya' clips in dataset: {existing_count}")
    print()

    print("Instructions:")
    print("1. Press ENTER to start a take.")
    print("2. Wait for '>>> SPEAK NOW: antriksya <<<'")
    print("3. Say the word clearly within 1.5 seconds.")
    print("4. Press Ctrl+C at any time to exit.")
    print("-" * 65)

    take_idx = 0
    while True:
        prompt = TAKE_PROMPTS[take_idx % len(TAKE_PROMPTS)]
        print(f"\n[Prompt #{take_idx + 1}] {prompt}")
        try:
            input("Press ENTER to record take (or Ctrl+C to finish)...")
        except (KeyboardInterrupt, EOFError):
            print("\nRecording session ended.")
            break

        print("  3...", end=" ", flush=True)
        time.sleep(0.4)
        print("2...", end=" ", flush=True)
        time.sleep(0.4)
        print("1...", flush=True)
        time.sleep(0.2)

        print("\n  >>> SPEAK NOW: antriksya <<<")
        audio_data = sd.rec(
            int(RECORD_WINDOW * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16'
        )
        sd.wait()
        audio_flat = audio_data.flatten()
        print("  [Done recording]")

        rms = np.sqrt(np.mean(audio_flat.astype(np.float32) ** 2))
        peak = np.max(np.abs(audio_flat))

        if peak < 400:
            print("  [!] Warning: Signal too quiet. Speak closer or louder. (Take discarded)")
            continue

        bounds = find_word_boundaries(audio_flat, threshold=max(500, int(peak * 0.18)))
        if bounds is None:
            print("  [!] Could not locate word boundary. (Take discarded)")
            continue

        start_s, end_s = bounds
        duration_ms = ((end_s - start_s) / SAMPLE_RATE) * 1000
        centered = center_clip(audio_flat, start_s, end_s)

        out_path, num = get_next_filename(OUTPUT_DIR)
        wavfile.write(out_path, SAMPLE_RATE, centered.astype(np.int16))

        print(f"  [+] Saved {os.path.basename(out_path)} | Word duration: {duration_ms:.0f}ms | Peak: {peak} | RMS: {rms:.1f}")
        take_idx += 1

    total_clips = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")])
    print(f"\nCompleted! Total 'antriksya' recordings: {total_clips}")
    print("Next step: Run `python augment_dataset.py` to synthesize 1,000+ training variations.")

if __name__ == "__main__":
    main()
