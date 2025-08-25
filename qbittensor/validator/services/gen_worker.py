#!/usr/bin/env python3
import time
import traceback
import torch
import numpy as np
import json
import os
import sys
import argparse
import logging
from datetime import datetime

log_file_handler = None

def setup_logging(args):
    """Sets up logging to redirect both print and bt.logging to a log file."""
    global log_file_handler
    log_filename = os.path.join(args.output_dir, f"worker_{os.getpid()}_seed_{args.seed}.log")
    log_file_handler = open(log_filename, 'w')

    sys.stdout = log_file_handler
    sys.stderr = log_file_handler
    
    print(f"--- Worker Log for PID {os.getpid()} ---")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Arguments: {vars(args)}")
    print("-" * 30)

    print("Configuring bittensor logger...")
    bt_logger = logging.getLogger()
    bt_logger.setLevel(logging.INFO)

    file_stream_handler = logging.StreamHandler(log_file_handler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_stream_handler.setFormatter(formatter)
    bt_logger.addHandler(file_stream_handler)
    print("Bittensor logger configured to write to this file.")


def main(args):
    """The main execution function for the worker."""
    setup_logging(args)
    try:
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    except Exception:
        pass

    from qbittensor.validator.peaked_circuit_creation.lib.circuit_gen import CircuitParams

    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    def _sync_cuda():
        if torch.cuda.is_available(): torch.cuda.synchronize()

    class Timer:
        def __init__(self): self.elapsed_ms = 0.0
        def __enter__(self):
            _sync_cuda()
            self._t0 = time.perf_counter()
            return self
        def __exit__(self, exc_type, exc, tb):
            _sync_cuda()
            self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0

    def safe_str(x) -> str:
        try: return str(x)
        except Exception: return f"<unprintable:{type(x).__name__}>"

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_id)
    except Exception as e:
        print(f"WARN: could not set CUDA device {args.gpu_id}: {e}")
    if PSUTIL_AVAILABLE and args.cpu_cores:
        p = psutil.Process()
        p.cpu_affinity(args.cpu_cores)

    with Timer() as t_total:
        params = CircuitParams.from_difficulty(args.difficulty)
        circuit = params.compute_circuit(args.seed)
        qasm = circuit.to_qasm()

    result = {
        "qubits": getattr(params, "nqubits", None),
        "rqc_depth": getattr(params, "rqc_depth", None),
        "pqc_depth": getattr(params, "pqc_depth", None),
        "total_ms": t_total.elapsed_ms,
        "qasm": qasm,
        "bitstring": safe_str(getattr(circuit, "target_state", None)),
        "peak_prob": getattr(circuit, "peak_prob", 0.0),
    }

    print("Saving artifacts...")
    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = f"seed_{args.seed}_rqc{result['rqc_depth']}_pqc{result['pqc_depth']}_diff{args.difficulty:.2f}"
    qasm_filename = f"{base_filename}.qasm"
    json_filename = f"{base_filename}_metadata.json"
    
    with open(os.path.join(args.output_dir, qasm_filename), 'w') as f:
        f.write(result['qasm'])

    print(f"Saving metadata to {json_filename}...")
    peak_prob = result['peak_prob']
    metadata = {
        "seed": args.seed,
        "difficulty": args.difficulty,
        "target_state": result['bitstring'],
        "peak_probability": peak_prob,
        "num_qubits": result['qubits'],
        "rqc_depth": result['rqc_depth'],
        "pqc_depth": result['pqc_depth'],
        "qasm_filename": qasm_filename,
        "num_shots": int(round(1 / peak_prob)) if peak_prob > 1e-12 else 0,
        "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(args.output_dir, json_filename), 'w') as f:
        json.dump(metadata, f, indent=4, cls=NumpyEncoder)
    # ---------------------------------------------------
    
    summary_data = {
        "seed": args.seed, "difficulty": args.difficulty, "qubits": result["qubits"],
        "rqc_depth": result["rqc_depth"], "pqc_depth": result["pqc_depth"],
        "total_ms": result["total_ms"],
    }
    sys.stdout = sys.__stdout__
    print(json.dumps(summary_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single circuit generation worker.", add_help=False)
    parser.add_argument("--difficulty", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument('--cpu-cores', type=lambda s: [int(item) for item in s.split(',')])
    args, unknown_args = parser.parse_known_args()
    
    try:
        main(args)
    except Exception as e:
        if log_file_handler:
            print("\n" + "="*50, file=log_file_handler)
            print("FATAL UNHANDLED EXCEPTION", file=log_file_handler)
            print("="*50, file=log_file_handler)
            traceback.print_exc(file=log_file_handler)
            log_file_handler.close()
        sys.exit(1)