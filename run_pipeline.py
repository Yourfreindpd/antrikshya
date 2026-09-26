"""
==============================================================================
ISRO SIH 26172: Complete End-to-End Pipeline Runner
==============================================================================
Orchestrates the entire KWS & ASR system:
1. Dataset Augmentation (augment_dataset.py)
2. TinyML Model Training (train_kws_model.py)
3. INT8 Quantization & Header Export (export_tflite_micro.py)
4. Automated Evaluation & Benchmarking (benchmark_metrics.py)
5. Live ASR Server & Web Dashboard (asr_server.py)
==============================================================================
"""

import sys
import subprocess
import os
import argparse

PYTHON_EXE = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_step(step_name, script_name, extra_args=None):
    print("\n" + "=" * 70)
    print(f"  >>> STEP: {step_name} ({script_name})")
    print("=" * 70)
    script_path = os.path.join(BASE_DIR, script_name)
    cmd = [PYTHON_EXE, script_path] + (extra_args or [])
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with exit code {res.returncode}.")
        sys.exit(res.returncode)
    print(f"\n[OK] Step '{step_name}' completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="ISRO SIH 26172 Pipeline Runner")
    parser.add_argument("--skip-augment", action="store_true", help="Skip dataset augmentation")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-export", action="store_true", help="Skip INT8 header export")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmarking")
    parser.add_argument("--skip-server", action="store_true", help="Skip starting the ASR server")
    args = parser.parse_args()

    print("======================================================================")
    print("        ISRO SIH 26172: END-TO-END PIPELINE ORCHESTRATOR")
    print("======================================================================")

    # Step 1: Augmentation
    if not args.skip_augment:
        run_step("1. Dataset Augmentation", "augment_dataset.py")

    # Step 2: Training
    if not args.skip_train:
        run_step("2. KWS Model Training", "train_kws_model.py")

    # Step 3: INT8 Header Export
    if not args.skip_export:
        run_step("3. INT8 Quantization & Header Export", "export_tflite_micro.py")

    # Step 4: Benchmarking
    if not args.skip_benchmark:
        run_step("4. Evaluation & Benchmarking", "benchmark_metrics.py")

    # # Step 5: ASR Server & Dashboard
    # if not args.skip_server:
    #     print("\n" + "=" * 70)
    #     print("  >>> STEP: 5. Live ASR Streaming Server & Dashboard")
    #     print("=" * 70)
    #     import socket
    #     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     port_in_use = sock.connect_ex(('127.0.0.1', 8000)) == 0
    #     sock.close()
    #     if port_in_use:
    #         print("[OK] ASR Server is already running and active on http://localhost:8000")
    #         print("[OK] ESP32 WebSocket endpoint is ready at ws://localhost:8000/ws/audio")
    #     else:
    #         print("Starting ASR Server on http://localhost:8000 ...")
    #         server_path = os.path.join(BASE_DIR, "asr_server.py")
    #         subprocess.run([PYTHON_EXE, server_path])

if __name__ == "__main__":
    main()
