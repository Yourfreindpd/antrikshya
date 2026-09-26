import os
import sys
import time
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
CHANNELS = 1
DURATION_SEC = 1

OUTPUT_DIR = r"C:\Users\yatha\Desktop\output"

# Number of samples in one recording
CLIP_SAMPLES = int(DURATION_SEC * SAMPLE_RATE)

# How long to let the stream "warm up" before trusting it
WARMUP_SEC = 1.5
WARMUP_SAMPLES = int(WARMUP_SEC * SAMPLE_RATE)

# Read from the stream in small chunks while warming up / flushing
CHUNK_SAMPLES = 1024


# ============================================================
# CHOOSE LABEL
# ============================================================

print("=" * 50)
print("        KEYWORD DATA COLLECTOR")
print("=" * 50)

print("\nChoose what you want to record:")
print("1 = keyword")
print("2 = unknown")
print("3 = silence")

choice = input("\nEnter choice: ").strip()

if choice == "1":
    label = "keyword"
elif choice == "2":
    label = "unknown"
elif choice == "3":
    label = "silence"
else:
    print("Invalid choice.")
    sys.exit()


# CREATE LABEL FOLDER

label_folder = os.path.join(OUTPUT_DIR, label)
os.makedirs(label_folder, exist_ok=True)


# FIND NEXT FILE NUMBER

existing_files = [
    f for f in os.listdir(label_folder)
    if f.endswith(".wav")
]

clip_count = len(existing_files)


# INSTRUCTIONS

print()
print("=" * 50)
print(f"Selected label: {label}")
print("=" * 50)

if label == "keyword":
    print("\nWhen recording starts:")
    print('Say your keyword ONCE, for example: "Jarvis"')
elif label == "unknown":
    print("\nWhen recording starts:")
    print("Say ONE word or short phrase that is NOT your keyword.")
elif label == "silence":
    print("\nWhen recording starts:")
    print("Stay completely quiet.")

print("\nPress ENTER to record.")
print("Press Ctrl+C to stop.")
print()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_n_samples(stream, n_samples):
    """
    Blocking-read exactly n_samples frames from an open InputStream,
    reading in chunks so we don't ask for a huge block in one call.
    Returns an (n_samples, CHANNELS) int16 numpy array.
    """
    collected = []
    remaining = n_samples

    while remaining > 0:
        to_read = min(CHUNK_SAMPLES, remaining)
        data, overflowed = stream.read(to_read)
        if overflowed:
            # Not fatal, but worth knowing about if it happens a lot
            print("[warning] input overflow while reading")
        collected.append(data.copy())
        remaining -= to_read

    return np.concatenate(collected, axis=0)


def flush_stream(stream):
    """
    Throw away any audio that has already piled up in the stream's
    internal buffer, so the next real recording starts from "now"
    instead of from stale/old audio.
    """
    available = stream.read_available
    if available > 0:
        stream.read(available)


# ============================================================
# OPEN MICROPHONE ONCE
# ============================================================

print("Opening microphone stream...")

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="int16",
    blocksize=CHUNK_SAMPLES,
)
stream.start()

print(f"Warming up for {WARMUP_SEC:.1f}s (discarding startup frames)...")
read_n_samples(stream, WARMUP_SAMPLES)

print("Stream stable. Ready to record.\n")


# ============================================================
# RECORDING LOOP
# ============================================================

try:
    while True:

        input("Press ENTER to record...")

        # Discard anything that accumulated in the buffer while we
        # were waiting for the user to press ENTER, so the clip we
        # capture starts right now, not a second ago.
        flush_stream(stream)

        print("\nRECORDING...")

        audio_data = read_n_samples(stream, CLIP_SAMPLES)

        print("RECORDING FINISHED.")

        # ========================================================
        # AUDIO CHECK
        # ========================================================

        rms = np.sqrt(
            np.mean(
                audio_data.astype(np.float32) ** 2
            )
        )

        print(
            f"RMS: {rms:.2f} | "
            f"Min: {audio_data.min()} | "
            f"Max: {audio_data.max()}"
        )

        # ========================================================
        # SAVE WAV
        # ========================================================

        file_path = os.path.join(
            label_folder,
            f"{label}_{clip_count:04d}.wav"
        )

        wavfile.write(
            file_path,
            SAMPLE_RATE,
            audio_data
        )

        print(f"Saved: {file_path}")

        clip_count += 1

        print()

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream.stop()
    stream.close()
    print("Microphone stream closed.")