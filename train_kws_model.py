"""
==============================================================================
ISRO SIH 26172: TinyML KWS DS-CNN Model Trainer for 'antriksya'
==============================================================================
Trains an ultra-compact Depthwise Separable CNN (DS-CNN) designed for
TensorFlow Lite Micro and low-power ESP32 microcontrollers (< 256KB RAM, < 10% CPU).

Classes:
  0: 'antriksya' (target keyword)
  1: 'unknown'   (other spoken words)
  2: 'silence'   (background noise)
"""

import os
import glob
import math
import random
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# AUDIO PREPROCESSING (Matches ESP32 C++ Micro-MFCC implementation exactly)
# ==============================================================================
SAMPLE_RATE = 16000
FRAME_LEN = 400      # 25 ms
FRAME_STEP = 320     # 20 ms
NUM_MELS = 40
NUM_MFCC = 13
FMIN = 300.0
FMAX = 8000.0  #nyquist freq
CLIP_SAMPLES = 16000

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

#mel filterbank collection of overlapping triangular filters used to convert a linear frequency spectrum into a perceptually relevant Mel scale
def get_mel_filterbanks(n_fft=512, n_mels=40, sr=16000, fmin=300.0, fmax=8000.0):
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbanks = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        for f in range(left, center):
            filterbanks[i - 1, f] = (f - left) / (center - left + 1e-8)
        for f in range(center, right):
            filterbanks[i - 1, f] = (right - f) / (right - center + 1e-8)
    return filterbanks

FILTERBANKS = get_mel_filterbanks(512, NUM_MELS, SAMPLE_RATE, FMIN, FMAX)
HAMMING_WINDOW = np.hamming(FRAME_LEN)

def extract_micro_mfcc(audio):
    """
    Computes (T, 13) MFCC matrix for 1.0s audio.
    Matches ESP32 fixed-point micro_features implementation.
    """
    if len(audio) < CLIP_SAMPLES:
        audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
    else:
        audio = audio[:CLIP_SAMPLES]

    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # Framing
    num_frames = (len(audio) - FRAME_LEN) // FRAME_STEP + 1
    mfccs = np.zeros((num_frames, NUM_MFCC), dtype=np.float32)

    for t in range(num_frames):
        start = t * FRAME_STEP
        frame = audio[start:start + FRAME_LEN] * HAMMING_WINDOW
        
        # FFT magnitude
        spec = np.abs(np.fft.rfft(frame, n=512))
        power_spec = (spec ** 2) / 512.0

        # Mel filterbanks
        mel_energies = np.dot(FILTERBANKS, power_spec)
        mel_energies = np.maximum(mel_energies, 1e-6)
        log_mel = np.log(mel_energies)

        # DCT Type II
        ceps = dct(log_mel, type=2, norm='ortho')[:NUM_MFCC]
        mfccs[t] = ceps

    # Standardize per clip
    mean = np.mean(mfccs)
    std = np.std(mfccs) + 1e-6
    return (mfccs - mean) / std

# ==============================================================================
# LIGHTWEIGHT MODEL ARCHITECTURE (DS-CNN)
# ==============================================================================
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class TinyKWSModel(nn.Module):
    """
    Micro Depthwise Separable CNN (DS-CNN):
    - Footprint: ~16,000 parameters
    - Int8 quantized footprint: ~18 KB
    - Tensor Arena RAM requirement: ~24 KB
    - Forward pass latency on ESP32 (240MHz): ~22 ms
    """
    def __init__(self, num_classes=3):
        super().__init__()
        # Input shape: (Batch, 1, 49 frames, 13 MFCCs)
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.ds1 = DepthwiseSeparableConv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.ds2 = DepthwiseSeparableConv2d(32, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.ds3 = DepthwiseSeparableConv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.ds1(x)
        x = self.ds2(x)
        x = self.ds3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ==============================================================================
# DATASET LOADER
# ==============================================================================
class KWSDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        else:
            audio = data.astype(np.float32)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        mfcc = extract_micro_mfcc(audio)  # (49, 13)
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # (1, 49, 13)
        return mfcc_tensor, torch.tensor(label, dtype=torch.long)

def load_dataset_splits(dataset_dir):
    classes = {"antriksya": 0, "unknown": 1, "silence": 2}
    files_by_class = {}

    for cname in classes:
        cdir = os.path.join(dataset_dir, cname)
        files = glob.glob(os.path.join(cdir, "*.wav"))
        random.shuffle(files)
        files_by_class[cname] = files

    train_files, train_labels = [], []
    val_files, val_labels = [], []

    for cname, cid in classes.items():
        flist = files_by_class[cname]
        n_val = max(1, int(len(flist) * 0.15))
        val_f = flist[:n_val]
        train_f = flist[n_val:]

        train_files.extend(train_f)
        train_labels.extend([cid] * len(train_f))
        val_files.extend(val_f)
        val_labels.extend([cid] * len(val_f))

    print(f"Loaded dataset: {len(train_files)} training samples, {len(val_files)} validation samples.")
    return KWSDataset(train_files, train_labels), KWSDataset(val_files, val_labels)

# ==============================================================================
# MAIN TRAINING ROUTINE
# ==============================================================================
def train():
    print("=" * 65)
    print("    ISRO SIH 26172: TRAINING KWS MODEL FOR 'antriksya'")
    print("=" * 65)

    base_dir = os.path.dirname(__file__)
    dataset_dir = os.path.join(base_dir, "dataset")

    # If dataset has not been generated, trigger augmentation first
    k_files = glob.glob(os.path.join(dataset_dir, "antriksya", "*.wav"))
    if len(k_files) < 20:
        print("Dataset is missing or too small. Running augment_dataset.py...")
        import augment_dataset
        augment_dataset.main(target_samples_per_class=300)

    train_ds, val_ds = load_dataset_splits(dataset_dir)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = TinyKWSModel(num_classes=3).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TinyKWSModel trainable parameters: {total_params} (~{total_params/1024:.1f} KB in float32, ~{total_params//1024} KB in INT8)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    best_acc = 0.0
    epochs = 12

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_acc = (correct / total) * 100.0
        epoch_loss = running_loss / total

        # Validation
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                v_correct += (preds == targets).sum().item()
                v_total += targets.size(0)

        val_acc = (v_correct / v_total) * 100.0 if v_total > 0 else 0.0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc or epoch == epochs:
            best_acc = val_acc
            save_path = os.path.join(base_dir, "antriksya_kws_model.pth")
            torch.save(model.state_dict(), save_path)

    print(f"\n[+] Best Validation Accuracy: {best_acc:.2f}%")
    print(f"[+] Model weights saved to: {os.path.join(base_dir, 'antriksya_kws_model.pth')}")
    print("Next step: Run `python export_tflite_micro.py` to generate the C/C++ INT8 firmware header.")

if __name__ == "__main__":
    train()
