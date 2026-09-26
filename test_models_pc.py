"""Run microphone audio or one WAV file through both models."""

import argparse
import os
import sys

import numpy as np
import sounddevice as sd
import torch
from scipy.io import wavfile

from export_tflite_micro import fuse_conv_bn, quantize_tensor_int8
from train_kws_model import TinyKWSModel, extract_micro_mfcc


BASE_DIR = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join(BASE_DIR, "antriksya_kws_model.pth")
CLASS_NAMES = ("antriksya", "unknown", "silence")


def load_audio(path):
    sample_rate, data = wavfile.read(path)
    if sample_rate != 16000:
        raise ValueError(f"{path} has sample rate {sample_rate}; expected 16000 Hz")
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    else:
        audio = data.astype(np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio


def record_microphone(duration):
    if duration <= 0:
        raise ValueError("Microphone duration must be greater than zero")
    sample_count = int(16000 * duration)
    print(f"Recording for {duration:.1f} seconds... speak now.")
    recording = sd.rec(sample_count, samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    return recording[:, 0]


def quantized_weight_model(float_model):
    """Build the same network using dequantized values of the exported INT8 weights."""
    model = TinyKWSModel(num_classes=3)

    with torch.no_grad():
        conv0, bn0 = float_model.init_conv[0], float_model.init_conv[1]
        w0, b0 = fuse_conv_bn(conv0, bn0)
        w0_int8, w0_scale = quantize_tensor_int8(w0)
        fused_conv0 = torch.nn.Conv2d(1, 16, (5, 3), stride=(2, 1), padding=(2, 1), bias=True)
        fused_conv0.weight.copy_(torch.from_numpy(w0_int8.astype(np.float32) * w0_scale))
        fused_conv0.bias.copy_(b0)
        model.init_conv = torch.nn.Sequential(fused_conv0, torch.nn.ReLU())

        for name in ("ds1", "ds2", "ds3"):
            source_block = getattr(float_model, name)
            target_block = getattr(model, name)

            depthwise_int8, depthwise_scale = quantize_tensor_int8(source_block.depthwise.weight)
            target_block.depthwise.weight.copy_(
                torch.from_numpy(depthwise_int8.astype(np.float32) * depthwise_scale)
            )

            pointwise_weight, pointwise_bias = fuse_conv_bn(
                source_block.pointwise, source_block.bn
            )
            pointwise_int8, pointwise_scale = quantize_tensor_int8(pointwise_weight)
            target_block.pointwise.weight.copy_(
                torch.from_numpy(pointwise_int8.astype(np.float32) * pointwise_scale)
            )
            target_block.pointwise.bias = torch.nn.Parameter(pointwise_bias.clone())
            target_block.bn.weight.fill_(1.0)
            target_block.bn.bias.zero_()
            target_block.bn.running_mean.zero_()
            target_block.bn.running_var.fill_(1.0)

        fc_int8, fc_scale = quantize_tensor_int8(float_model.fc.weight)
        model.fc.weight.copy_(torch.from_numpy(fc_int8.astype(np.float32) * fc_scale))
        model.fc.bias.copy_(float_model.fc.bias)

    return model.eval()


def probabilities(model, features):
    with torch.no_grad():
        logits = model(torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0))
        return torch.softmax(logits, dim=1)[0].numpy()


def print_result(title, values):
    prediction = int(np.argmax(values))
    print(f"\n{title}")
    for name, probability in zip(CLASS_NAMES, values):
        print(f"  {name:10s}: {probability:.6f} ({probability * 100:6.2f}%)")
    print(f"  prediction : {CLASS_NAMES[prediction]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", help="16 kHz WAV file to classify")
    parser.add_argument(
        "--mic",
        type=float,
        metavar="SECONDS",
        default=1.0,
        help="length of each continuous microphone window in seconds (default: 1)",
    )
    args = parser.parse_args()

    if args.audio is not None and "--mic" in sys.argv:
        parser.error("provide either an audio file or --mic, not both")

    audio_path = args.audio
    if args.audio is not None:
        audio = load_audio(audio_path)
        audio_label = os.path.abspath(audio_path)

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Missing trained model: {WEIGHTS_PATH}")

    float_model = TinyKWSModel(num_classes=3)
    float_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    float_model.eval()
    quantized_model = quantized_weight_model(float_model)

    if args.audio is not None:
        features = extract_micro_mfcc(audio).astype(np.float32)
        print(f"Audio: {audio_label}")
        print(f"MFCC shape: {features.shape} | classes: {list(CLASS_NAMES)}")
        print_result("PyTorch float32 model", probabilities(float_model, features))
        print_result("INT8-weight simulation", probabilities(quantized_model, features))
        return

    print("Continuous microphone mode. Press Ctrl+C to stop.")
    try:
        while True:
            audio = record_microphone(args.mic)
            features = extract_micro_mfcc(audio).astype(np.float32)
            print(f"MFCC shape: {features.shape} | classes: {list(CLASS_NAMES)}")
            print_result("PyTorch float32 model", probabilities(float_model, features))
            print_result("INT8-weight simulation", probabilities(quantized_model, features))
    except KeyboardInterrupt:
        print("\nStopped microphone mode.")


if __name__ == "__main__":
    main()