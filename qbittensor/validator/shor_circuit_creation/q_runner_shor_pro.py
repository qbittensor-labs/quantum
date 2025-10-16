import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, Tuple, Optional, Any, Iterable

from qiskit.qasm2 import load as qasm2_load, loads as qasm2_loads
from qiskit_aer import AerSimulator

# regex helpers
_MEASURE_RE = re.compile(r"measure\s+([A-Za-z_]\w*)\[(\d+)\]\s*->\s*([A-Za-z_]\w*)\[(\d+)\]\s*;")
_DECL_RE    = re.compile(r'^\s*(qreg|creg)\s+([A-Za-z_]\w*)\s*\[(\d+)\]\s*;', flags=re.M)

# Phase register name heuristics
_PHASE_QREG_CANDIDATES = {"ph", "phase"}
_PHASE_CREG_CANDIDATES = {"pc", "c"}


# small utils

def die(msg: str, code: int = 1) -> None:
    print(f"[fatal] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)

def read_file(path: str) -> str:
    if not path:
        die("No QASM path provided.")
    if not os.path.exists(path):
        die(f"File not found: {path}")
    with open(path, "r") as f:
        return f.read()

def write_txt_counts(path: str, counts: Dict[str, int]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))  # by count desc, then lexicographic
    with open(path, "w") as f:
        for bitstr, c in items:
            if bitstr:
                f.write(f"{bitstr} {int(c)}\n")

def has_measures(qc) -> bool:
    try:
        if getattr(qc, "num_clbits", 0) > 0:
            return True
    except Exception:
        pass
    try:
        ops = qc.count_ops()
        return bool(ops.get("measure", 0))
    except Exception:
        return False

def _stamp(label: str, t0: float, last: float) -> float:
    now = time.perf_counter()
    print(f"[profile] {label}: {now - last:.3f}s (cum {now - t0:.3f}s)", flush=True)
    return now

def _try_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


_SWAP_LINE_RE = re.compile(r'^\s*swap\s+([A-Za-z_]\w*\[\d+\])\s*,\s*([A-Za-z_]\w*\[\d+\])\s*;\s*$', flags=re.M)

def _expand_swap_to_cx(qasm: str) -> Tuple[str, int]:
    """
    Replace each line "swap a[i],b[j];" with the 3-CX pattern.
    Returns (new_text, num_replacements).
    """
    count = 0
    def _one(m: re.Match) -> str:
        nonlocal count
        a, b = m.group(1), m.group(2)
        count += 1
        return f"cx {a},{b};\ncx {b},{a};\ncx {a},{b};"
    new = _SWAP_LINE_RE.sub(_one, qasm)
    return new, count


def parse_creg_sizes(qasm_text: str) -> Dict[str, int]:
    sizes = {}
    for kind, name, size in _DECL_RE.findall(qasm_text):
        if kind == "creg":
            sizes[name] = int(size)
    return sizes

def parse_phase_mapping_from_qasm_text(qasm_text: str) -> Tuple[Optional[int], Dict[int, int], Optional[str]]:
    """
    Returns:
      K: int or None, width of the destination classical register for phase bits
      phase_to_c: dict {i -> j} for 'measure <phase_qreg>[i] -> <creg>[j];'
      c_name: str or None, classical register receiving phase bits
    Heuristics: prefer phase qreg among {'ph','phase'}; otherwise pick the (qreg,creg)
    pair with the most measure lines, preferring creg in {'pc','c'}.
    """
    creg_sizes = parse_creg_sizes(qasm_text)

    preferred_hits = []
    generic_hits   = []
    for m in _MEASURE_RE.finditer(qasm_text):
        qreg, i_str, creg, j_str = m.groups()
        i, j = int(i_str), int(j_str)
        hit = (qreg, i, creg, j)
        (preferred_hits if qreg in _PHASE_QREG_CANDIDATES else generic_hits).append(hit)

    hits = preferred_hits
    if not hits and generic_hits:
        tally = {}
        for qreg, i, creg, j in generic_hits:
            key = (qreg, creg)
            tally[key] = tally.get(key, 0) + 1
        cand = [k for k in tally if k[1] in _PHASE_CREG_CANDIDATES]
        key = max(cand, key=lambda k: tally[k]) if cand else max(tally, key=lambda k: tally[k])
        qreg_pick, creg_pick = key
        hits = [(q, i, c, j) for (q, i, c, j) in generic_hits if q == qreg_pick and c == creg_pick]

    if not hits:
        return None, {}, None

    phase_to_c: Dict[int,int] = {}
    detected_c = None
    for qreg, i, creg, j in hits:
        phase_to_c[i] = j
        detected_c = creg

    K = creg_sizes.get(detected_c, (max(phase_to_c.values()) + 1) if phase_to_c else None)
    return K, phase_to_c, detected_c

def normalize_backend_bits_msb_left(bits: str, *, K: int) -> str:
    s = bits.strip().replace(" ", "")
    if len(s) < K:
        s = ("0" * (K - len(s))) + s
    return s[:K]

def embed_phase_bits_full(full_bits: str, K: int, phase_to_c_map: Dict[int, int]) -> str:
    """
    Build a full-length MSB-left string of size K (classical register width),
    filling only measured phase positions. c[j] maps to position (K-1-j).
    """
    s = normalize_backend_bits_msb_left(full_bits, K=K)
    arr = ["0"] * K
    for _, j in phase_to_c_map.items():
        pos = K - 1 - j
        if 0 <= pos < len(s):
            arr[pos] = s[pos]
    return "".join(arr)


def build_aer_backend(name: str) -> AerSimulator:
    key = (name or "").lower()
    if key in {"mps", "aer_mps"}:
        return AerSimulator(method="matrix_product_state")
    if key in {"statevector", "aer_statevector", "sv"}:
        return AerSimulator(method="statevector")
    if key in {"dm", "density_matrix"}:
        return AerSimulator(method="density_matrix")
    if key in {"stabilizer", "aer_stabilizer"}:
        return AerSimulator(method="stabilizer")
    try:
        return AerSimulator(method=key)
    except Exception as e:
        die(f"Unsupported backend '{name}': {e}")
        raise  # unreachable


# MPS metadata summarizer

def _flatten_numbers(obj: Any) -> Iterable[float]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _flatten_numbers(v)
    elif isinstance(obj, (int, float)):
        yield float(obj)

def _summarize_mps_metadata(meta: Dict[str, Any]) -> None:
    """
    Print a compact MPS summary if log data is present:
    - max bond (approx: max int seen in 'bond' fields or any numeric)
    - total truncated/discarded weight (sum of small floats)
    Structure varies by Aer version, so we try to be robust.
    """
    mps = meta.get("MPS_log_data") or meta.get("mps_log_data") or meta.get("mps") or {}
    if not mps:
        print("[mps] no MPS_log_data in metadata", flush=True)
        return

    # Heuristic scan
    max_bond = 0
    total_discard = 0.0

    def _scan(k: str, v: Any):
        nonlocal max_bond, total_discard
        kl = k.lower()
        if isinstance(v, (list, tuple, dict)):
            if isinstance(v, dict):
                for kk, vv in v.items():
                    _scan(kk, vv)
            else:
                for vv in v:
                    _scan(k, vv)
        elif isinstance(v, (int, float)):
            if "bond" in kl:
                try:
                    max_bond = max(max_bond, int(v))
                except Exception:
                    pass
            if ("trunc" in kl) or ("discard" in kl):
                try:
                    total_discard += float(v)
                except Exception:
                    pass

    if isinstance(mps, dict):
        for k, v in mps.items():
            _scan(k, v)
    else:
        nums = list(_flatten_numbers(mps))
        if nums:
            max_bond = int(max(nums))
            total_discard = sum(x for x in nums if 0 <= x < 1e-3)

    print(f"[mps] summary: max_bond≈{max_bond}  total_discarded≈{total_discard:.2e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Run OpenQASM 2.0 on Qiskit Aer (selectable backend).")
    ap.add_argument("qasm", help="Path to .qasm (OpenQASM 2.0)")
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--backend", default="mps",
                    help="Aer method: mps|statevector|dm|stabilizer (default: mps)")
    ap.add_argument("--print-ops", action="store_true", help="Print operation counts before running")

    # TXT counts
    ap.add_argument("--txt-out", default=None,
                    help="Write counts '<bitstring> <count>' (default: results/<base>.counts.txt)")

    ap.add_argument("--out", default=None, help="Optional Top-K JSON path")
    ap.add_argument("--topk-out", type=int, default=100, help="How many top bitstrings to keep in JSON (default:100)")
    ap.add_argument("--include-raw", action="store_true", help="Also include full raw counts in the JSON (large)")

    ap.add_argument("--profile", action="store_true", help="Print coarse phase timings")
    ap.add_argument("--profile-cpu", action="store_true", help="Sample CPU%% during execution (requires psutil)")

    ap.add_argument("--mps-max-bond", type=int, default=None,
                    help="Max bond dimension for MPS (matrix_product_state_max_bond_dimension). Default: unlimited.")
    ap.add_argument("--mps-threshold", type=float, default=None,
                    help="Truncation threshold for MPS SVD (matrix_product_state_truncation_threshold). "
                         "Use 0 to disable truncation. Default: Aer's default (typically very small).")
    ap.add_argument("--mps-log", action="store_true",
                    help="Enable MPS logging (mps_log_data=True) and print a summary after the run.")

    args = ap.parse_args()

    print("===== Qiskit QASM Runner (Aer) =====", flush=True)

    t0 = time.perf_counter()

    # Read QASM and validate header
    qasm_text_from_disk = read_file(args.qasm)
    header = qasm_text_from_disk.lstrip().splitlines()[:1]
    if not header:
        die("QASM file is empty.")
    if not header[0].strip().upper().startswith("OPENQASM 2.0"):
        die("This runner expects OpenQASM 2.0.")

    # Parse phase mapping from QASM text (names: ph/pc or phase/c)
    K_phase, phase_to_c, phase_creg = (None, {}, None)
    try:
        K_phase, phase_to_c, phase_creg = parse_phase_mapping_from_qasm_text(qasm_text_from_disk)
    except Exception as e:
        print(f"[map] phase mapping parse failed: {e}", flush=True)
    t_read = _stamp("read+parse_map", t0, t0) if args.profile else None

    # Load QuantumCircuit from QASM2
    # Try a clean load first; if 'swap' isn't defined for some builds, expand swaps and loads() from string.
    try:
        qc = qasm2_load(args.qasm)
    except Exception:
        fixed, n_swaps = _expand_swap_to_cx(qasm_text_from_disk)
        if n_swaps:
            print(f"[normalize] expanded {n_swaps} swap -> cx,cx,cx (temp load)", flush=True)
        qc = qasm2_loads(fixed)
    t_load = _stamp("qasm2.load(s)", t0, t_read or t0) if args.profile else None

    # Basic circuit info
    try:
        nq = qc.num_qubits
    except Exception:
        nq = "?"
    print(f"[circuit] qubits={nq}", flush=True)

    if args.print_ops:
        try:
            ops = qc.count_ops()
            total_ops = sum(int(v) for v in ops.values())
            cx = int(ops.get("cx", 0))
            cz = int(ops.get("cz", 0))
            sw = int(ops.get("swap", 0))
            twoq = cx + cz + sw
            print(f"[ops] total: {total_ops}  twoq≈{twoq}  cx={cx} cz={cz} swap={sw}", flush=True)
            for k, v in sorted(ops.items(), key=lambda kv: kv[1], reverse=True):
                print(f"   {k:8s}: {v}", flush=True)
        except Exception:
            print("[ops] count_ops() not available.", flush=True)
    t_ops = _stamp("count_ops", t0, t_load or t0) if args.profile else None

    # Ensure measurements only if missing
    if not has_measures(qc):
        try:
            qc.measure_all()
            print("[measure] added measure_all()", flush=True)
        except Exception:
            print("[measure] measure_all() unavailable; proceeding as-is", flush=True)
    t_measure = _stamp("measure_patch", t0, t_ops or t_load or t0) if args.profile else None

    # Build backend
    backend = build_aer_backend(args.backend)

    # Apply MPS options if applicable
    if getattr(backend.options, "method", None) == "matrix_product_state":
        mps_opts = {}
        if args.mps_max_bond is not None:
            mps_opts["matrix_product_state_max_bond_dimension"] = args.mps_max_bond
        if args.mps_threshold is not None:
            mps_opts["matrix_product_state_truncation_threshold"] = float(args.mps_threshold)
        if args.mps_log:
            mps_opts["mps_log_data"] = True
        if mps_opts:
            backend.set_options(**mps_opts)

    try:
        backend.set_options(coupling_map=None)
    except Exception:
        pass

    print(f"[backend] {backend}", flush=True)
    try:
        method_str = backend.options.method
    except Exception:
        method_str = "?"
    print(f"[run] engine=AerSimulator(method={method_str}) shots={args.shots}", flush=True)

    # Optional CPU sampler
    psutil = _try_import_psutil() if args.profile_cpu else None
    sampler_thread = None
    cpu_series = []
    if psutil:
        import threading
        def _sample_cpu():
            psutil.cpu_percent(interval=None)  # prime
            while sampler_thread is not None:
                cpu_series.append(psutil.cpu_percent(interval=0.2))
        sampler_thread = threading.Thread(target=_sample_cpu, daemon=True)
        sampler_thread.start()
        print("[profile] CPU sampler started", flush=True)
    elif args.profile_cpu:
        print("[profile] psutil not available; skipping CPU sampling", flush=True)

    # Run — submit the circuit AS-IS and DISABLE validation/transpile paths
    t_submit = time.perf_counter()
    job = backend.run(qc, shots=args.shots, validate=False)
    result = job.result()  # block
    t_done = time.perf_counter()

    if sampler_thread is not None:
        tmp = sampler_thread
        sampler_thread = None
        tmp.join(timeout=1.0)
        if cpu_series:
            avg_cpu = sum(cpu_series)/len(cpu_series)
            p95 = sorted(cpu_series)[int(0.95*len(cpu_series))-1]
            print(f"[profile] CPU during run avg={avg_cpu:.1f}% p95={p95:.1f}% samples={len(cpu_series)}", flush=True)

    if args.profile:
        print(f"[profile] run() + result: {t_done - t_submit:.3f}s", flush=True)
        print(f"[profile] total: {t_done - t0:.3f}s", flush=True)

    elapsed = t_done - t_submit
    print(f"[time] run elapsed: {elapsed:.3f}s", flush=True)

    # Fetch raw counts from backend
    counts_raw = result.get_counts()
    if isinstance(counts_raw, list) and counts_raw:
        counts_raw = counts_raw[0]
    total_shots = sum(int(v) for v in counts_raw.values())
    print(f"[result] unique_outcomes={len(counts_raw)} total_shots={total_shots}", flush=True)

    # Optional: dump MPS metadata summary
    if args.mps_log:
        try:
            md = result.results[0].metadata  # first (and only) circuit
            _summarize_mps_metadata(md)
        except Exception as e:
            print(f"[mps] could not read metadata: {e}", flush=True)

    # Phase-only histogram (full-width) if mapping available; else raw
    if phase_to_c and K_phase and phase_creg:
        phase_hist = Counter()
        for full_bits, cnt in counts_raw.items():
            b_full = embed_phase_bits_full(full_bits, K_phase, phase_to_c)
            phase_hist[b_full] += int(cnt)
        counts_out = dict(phase_hist)
        print(f"[map] phase-only export: width_set=[{K_phase}] (from creg {phase_creg}[{K_phase}])", flush=True)
    else:
        counts_out = {str(k).replace(" ", ""): int(v) for k, v in counts_raw.items()}
        print("[map] no phase mapping found; exporting raw counts", flush=True)

    # Write TXT counts
    base = os.path.splitext(os.path.basename(args.qasm))[0]
    txt_out = args.txt_out or os.path.join("results", f"{base}.counts.txt")
    os.makedirs(os.path.dirname(txt_out) or ".", exist_ok=True)
    write_txt_counts(txt_out, counts_out)
    print(f"[ok] wrote TXT counts: {txt_out}", flush=True)

    # Optional Top-K JSON
    if args.out:
        topk = max(1, int(args.topk_out))
        top = sorted(counts_out.items(), key=lambda kv: kv[1], reverse=True)[:topk]
        payload = {
            "meta": {
                "qasm_file": os.path.basename(args.qasm),
                "backend": f"AerSimulator(method={method_str})",
                "qubits": int(qc.num_qubits) if hasattr(qc, "num_qubits") else None,
                "shots": args.shots,
                "classical_width": K_phase if phase_to_c else None,
                "phase_bits": len(phase_to_c) if phase_to_c else None,
                "mps_max_bond": args.mps_max_bond,
                "mps_threshold": args.mps_threshold,
            },
            "timing": {"elapsed_seconds": elapsed},
            "summary": {"total_shots": total_shots, "unique_outcomes": len(counts_out), "top_k": topk},
            "hist": {b: int(c) for b, c in top}
        }
        if args.include_raw:
            payload["counts"] = counts_out
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"[ok] wrote Top-{topk} JSON: {args.out}", flush=True)

    print("===== done =====", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        die(f"Unhandled exception: {e}")
