#!/usr/bin/env python3
"""
q_generator_shor_pro.py

tuned for Qiskit 2.x runners (Aer MPS etc.).

CLI

  --level Lk
  --rng-seed S
  --emit-json
  --outdir PATH
  --k-list "k1,k2,..."
  --k-range lo:hi
  --beacon HEX --miner-id STR --variants K --recommended-shots M --deadline-minutes D
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import secrets
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.qasm2 import dumps as qasm2_dumps


# verifier bindings helpers

import hashlib

def _msb_to_lsb_positions(indices, t):
    """MSB-left print indices (0..t-1) -> LSB-0 physical positions used in bit ops."""
    return [t - 1 - int(i) for i in indices]

def _normalize_window_positions(logical_list, log2phys_map, t, indexing="msb"):
    """
    Build bit positions the verifier will pack from the t-bit integer.
    """
    if not logical_list:
        return []
    if log2phys_map:
        phys = [int(log2phys_map.get(int(L), int(L))) for L in logical_list]
    else:
        phys = _msb_to_lsb_positions(logical_list, t) if (indexing or "msb").lower()=="msb" else [int(L) for L in logical_list]
    return sorted(int(p) for p in phys)

def _pack_bits_from_positions(x, positions):
    """Pack selected bit positions of x into a compact integer"""
    v = 0
    for i, p in enumerate(positions):
        v |= ((int(x) >> int(p)) & 1) << i
    return v

def _attach_verifier_bindings(meta):
    """
    Enrich meta_private.json 
    """
    t = int(meta["t"])
    meta.setdefault("phase_indexing", "msb")

    # top-level canonical
    k_top = list(meta.get("secret", {}).get("selection", {}).get("k_list_canonical", []))
    if k_top:
        meta["phase_logical_to_physical"] = {str(int(L)): (t - 1 - int(L)) for L in k_top}

    delta_t = int(meta["secret"]["nonce_delta"])

    for v in meta["variants"]:
        k_log = list(v.get("k_list", [])) or k_top
        # logical->physical map (LSB-0)
        if not v.get("phase_logical_to_physical"):
            v["phase_logical_to_physical"] = {str(int(L)): (t - 1 - int(L)) for L in k_log}
        v["phase_indexing"] = v.get("phase_indexing", meta.get("phase_indexing", "msb"))
        phys_positions = _normalize_window_positions(
            k_log, v.get("phase_logical_to_physical", {}), t, indexing=v["phase_indexing"]
        )
        W = len(phys_positions)
        delta_w = _pack_bits_from_positions(delta_t, phys_positions) & ((1 << W) - 1) if W > 0 else 0
        v["delta_w"] = int(delta_w)
        v["k_fingerprint"] = hashlib.blake2b(
            ("|".join(map(str, k_log))).encode("utf-8"), digest_size=8
        ).hexdigest()

    return meta


def _seed_from_protocol(beacon_hex: Optional[str], miner_id: Optional[str], fallback: Optional[int]) -> int:
    """Seed derived from (beacon, miner_id) in protocol mode, else fallback/random."""
    if beacon_hex and miner_id:
        h = hashlib.blake2b(digest_size=16)
        h.update(bytes.fromhex(beacon_hex))
        h.update(miner_id.encode("utf-8"))
        return int.from_bytes(h.digest(), "big")
    return seed_all(fallback)

def seed_all(rng_seed: Optional[int]) -> int:
    if rng_seed is None:
        rng_seed = random.randrange(1 << 63)
    random.seed(rng_seed)
    return rng_seed

def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text)
        tmp.replace(path)
    except Exception:
        path.write_text(text)

def _write_qasm(qc: "QuantumCircuit", path: Path) -> None:
    # Export QASM2 and ensure std gates are in scope
    qasm_str = qasm2_dumps(qc)
    if 'include "qelib1.inc";' not in qasm_str:
        lines = qasm_str.splitlines()
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("openqasm 2.0"):
                lines.insert(i + 1, 'include "qelib1.inc";')
                qasm_str = "\n".join(lines)
                break
    _atomic_write_text(path, qasm_str)

def _atomic_write_qasm(qc: "QuantumCircuit", path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        _write_qasm(qc, tmp)
        tmp.replace(path)
    except Exception:
        _write_qasm(qc, path)

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def qasm_sha256_file(path: Path) -> str:
    return sha256_hex(path.read_bytes())

# salts, k windows, compaction

def derive_salt(beacon_hex: str, miner_id: str, idx: int) -> str:
    return sha256_hex(f"QR|v1|{beacon_hex}|{miner_id}|{idx}".encode())

def klist_variant_from_salt(salt_hex: str, t: int, base_len: int, idx: Optional[int] = None) -> List[int]:
    """
    Choose a contiguous window of width base_len in [0..t-1].
    """
    w = min(max(1, base_len), t)
    if w >= t:
        return list(range(t))
    max_shift = t - w
    rng = random.Random(int(salt_hex[:32], 16))
    start = rng.randrange(0, max_shift + 1)
    return list(range(start, start + w))

def build_phase_mapping(t: int, k_list: List[int]) -> Dict[str, object]:
    """
    Map logical window ks to physical [0..w-1] and park remaining logical bits at [w..t-1].
    Returns: {log2phys, phys2log, phys_window, phys_nonwindow, w, t, k_sorted}
    """
    k_sorted = sorted(set(int(k) for k in k_list))
    w = len(k_sorted)
    rest = [k for k in range(t) if k not in k_sorted]
    phys2log = k_sorted + rest
    log2phys = {log: phys for phys, log in enumerate(phys2log)}
    return {
        "log2phys": log2phys,
        "phys2log": phys2log,
        "phys_window": list(range(0, w)),
        "phys_nonwindow": list(range(w, t)),
        "w": w,
        "t": t,
        "k_sorted": k_sorted,
    }


# Number theory & exact order synthesis (ord_N(a) = r)

def v2(x: int) -> int:
    c = 0
    while x % 2 == 0:
        x //= 2
        c += 1
    return c

def is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11, 13, 17]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def prime_factors(n: int) -> Dict[int, int]:
    f: Dict[int, int] = {}
    x = n
    while x % 2 == 0:
        f[2] = f.get(2, 0) + 1
        x //= 2
    p = 3
    while p * p <= x:
        while x % p == 0:
            f[p] = f.get(p, 0) + 1
            x //= p
        p += 2
    if x > 1:
        f[x] = f.get(x, 0) + 1
    return f

def crt_pair(a1: int, m1: int, a2: int, m2: int) -> int:
    inv = pow(m1, -1, m2)
    return (a1 + m1 * ((a2 - a1) * inv % m2)) % (m1 * m2)

def balanced_partition_prime_powers(r: int) -> Tuple[int, int]:
    parts = sorted([p ** e for p, e in prime_factors(r).items()], reverse=True)
    rp, rq = 1, 1
    for v in parts:
        if rp <= rq:
            rp *= v
        else:
            rq *= v
    assert (rp * rq) // math.gcd(rp, rq) == r
    return rp, rq

def find_prime_congruent_to_one(
    modulus: int, *, target_bits: Optional[int] = None, rng: Optional[random.Random] = None,
    max_k: int = 2_000_000, time_budget_s: float = 4.0
) -> int:
    if rng is None:
        rng = random
    t0 = time.time()

    def timed_out() -> bool:
        return (time.time() - t0) > time_budget_s

    def ok(p: int) -> bool:
        return (p & 1) and is_probable_prime(p)

    if target_bits is None:
        for k in range(1, 20000):
            if timed_out():
                break
            p = k * modulus + 1
            if ok(p):
                return p
        tries = 0
        while tries < 10000 and not timed_out():
            k = rng.randint(1, max_k)
            p = k * modulus + 1
            if ok(p):
                return p
            tries += 1
        raise RuntimeError("timeout")
    low = (1 << (target_bits - 1)) + 1
    high = (1 << target_bits) - 1
    k_min = (low - 1) // modulus + 1
    k_max = (high - 1) // modulus
    if k_min > k_max:
        tb = max(target_bits, (modulus + 1).bit_length() + 1)
        return find_prime_congruent_to_one(modulus, target_bits=tb, rng=rng, max_k=max_k, time_budget_s=time_budget_s)
    ks = list(range(k_min, k_max + 1))
    edge = min(50000, len(ks))
    for k in ks[:edge]:
        if timed_out():
            break
        p = k * modulus + 1
        if ok(p):
            return p
    tries = 0
    while tries < 20000 and not timed_out():
        k = ks[rng.randrange(len(ks))]
        p = k * modulus + 1
        if ok(p):
            return p
        tries += 1
    for bump in (1, 2, 3, 4):
        tb = max(target_bits + bump, (modulus + 1).bit_length() + 1)
        try:
            return find_prime_congruent_to_one(modulus, target_bits=tb, rng=rng, max_k=max_k, time_budget_s=max(0.5, time_budget_s / 2))
        except RuntimeError:
            continue
    raise RuntimeError("timeout near target_bits")

def synthesize_semiprime_and_base_for_period(
    r: int, *, seed: Optional[int] = None, target_n_bits: Optional[int] = None
) -> Tuple[int, int, int, int, int, int]:
    if r < 2:
        raise ValueError("r >= 2 required")
    rp, rq = balanced_partition_prime_powers(r)
    if target_n_bits is not None:
        half = max(2, target_n_bits // 2)
        min_p = (rp + 1).bit_length()
        min_q = (rq + 1).bit_length()
        p_bits = max(min_p, half)
        q_bits = max(min_q, half)
    else:
        p_bits = max(2, (rp + 1).bit_length())
        q_bits = max(2, (rq + 1).bit_length())

    rng_local = random.Random(seed)

    def prim_root(mod: int) -> int:
        phi = mod - 1
        pf = prime_factors(phi)
        for g in range(2, mod - 1):
            if all(pow(g, phi // qf, mod) != 1 for qf in pf):
                return g
        raise RuntimeError("no primitive root")

    for _ in range(4):
        p_scan = [p_bits + d for d in (0, +1, -1, +2, -2, +3, -3) if p_bits + d >= (rp + 1).bit_length()]
        q_scan = [q_bits + d for d in (0, +1, -1, +2, -2, +3, -3) if q_bits + d >= (rq + 1).bit_length()]
        for pb in p_scan:
            for qb in q_scan:
                p = find_prime_congruent_to_one(rp, target_bits=pb, rng=rng_local, time_budget_s=4.0)
                q = find_prime_congruent_to_one(rq, target_bits=qb, rng=rng_local, time_budget_s=4.0)
                if p == q:
                    continue
                N = p * q
                g_p = prim_root(p)
                g_q = prim_root(q)
                a_p = pow(g_p, (p - 1) // rp, p)
                a_q = pow(g_q, (q - 1) // rq, q)
                a = crt_pair(a_p, p, a_q, q)
                if pow(a, r, N) != 1:
                    continue
                ok = True
                for tprime in set(prime_factors(r).keys()):
                    if pow(a, r // tprime, N) == 1:
                        ok = False
                        break
                if not ok:
                    continue
                return p, q, N, a, rp, rq
    raise RuntimeError("synthesis failed")


# Adders / modular ops

def toffoli_std(qc, c1, c2, t):
    qc.h(t); qc.cx(c2, t); qc.tdg(t); qc.cx(c1, t); qc.t(t); qc.cx(c2, t); qc.tdg(t)
    qc.cx(c1, t); qc.t(c2); qc.t(t); qc.h(t); qc.cx(c1, c2); qc.t(c1); qc.tdg(c2); qc.cx(c1, c2)

def toffoli_relphase(qc, c1, c2, t):
    qc.h(t); qc.cx(c2, t); qc.tdg(t); qc.cx(c1, t); qc.t(t); qc.cx(c2, t); qc.tdg(t)
    qc.cx(c1, t); qc.t(c2); qc.t(t); qc.h(t)

def _bit_index(j: int, n: int, ripple_direction: str) -> int:
    return j

def toffoli_add(qc, c1, c2, t):
    toffoli_relphase(qc, c1, c2, t)

def maj_const_sel(qc, kbit: int, sel, b, c):
    if kbit:
        qc.cx(sel, b)
    qc.cx(b, c)
    if kbit:
        qc.cx(sel, c)
        toffoli_add(qc, sel, b, c)

def uma_const_sel(qc, kbit: int, sel, b, c):
    if kbit:
        toffoli_add(qc, sel, b, c)
        qc.cx(sel, c)
    qc.cx(b, c)
    if kbit:
        qc.cx(sel, b)

def add_const_inplace_rc_sel(qc, sel, X, K_val: int, carry, *, ripple_direction: str):
    n = len(X)
    for jj in range(n):
        j = _bit_index(jj, n, ripple_direction)
        kpos = (n - 1 - j)
        kbit = (K_val >> kpos) & 1
        maj_const_sel(qc, kbit, sel, X[j], carry)
    for jj in reversed(range(n)):
        j = _bit_index(jj, n, ripple_direction)
        kpos = (n - 1 - j)
        kbit = (K_val >> kpos) & 1
        uma_const_sel(qc, kbit, sel, X[j], carry)

def add_const_modN_rc_sel_inplace(qc, sel, X, K_val: int, N: int, sel_one, carry, flag, *, ripple_direction: str):
    n = len(X)
    Nbar = ((1 << n) - 1) ^ N
    add_const_inplace_rc_sel(qc, sel, X, K_val, carry, ripple_direction=ripple_direction)
    qc.x(carry)
    for jj in range(n):
        j = _bit_index(jj, n, ripple_direction)
        kpos = (n - 1 - j)
        kbit = (Nbar >> kpos) & 1
        maj_const_sel(qc, kbit, sel_one, X[j], carry)
    qc.x(carry)
    toffoli_std(qc, sel, carry, flag)
    qc.x(carry)
    add_const_inplace_rc_sel(qc, flag, X, N, carry, ripple_direction=ripple_direction)
    qc.x(carry)
    toffoli_std(qc, sel, carry, flag)
    qc.x(carry)
    for jj in reversed(range(n)):
        j = _bit_index(jj, n, ripple_direction)
        kpos = (n - 1 - j)
        kbit = (Nbar >> kpos) & 1
        uma_const_sel(qc, kbit, sel_one, X[j], carry)

def cswap(qc: "QuantumCircuit", ctrl, a, b):
    toffoli_std(qc, ctrl, b, a)
    toffoli_std(qc, ctrl, a, b)
    toffoli_std(qc, ctrl, b, a)

def modmul_const_rc_controlled_true_inplace(
    qc, ctrl, X, Y, sel, sel_one, carry, flag, a: int, N: int, *, ripple_direction: str
):
    if a % N == 1:
        return
    n = len(X)
    for jj in range(n):
        j = _bit_index(jj, n, ripple_direction)
        weight = pow(2, n - 1 - j, N)
        Kj = (a * weight) % N
        if Kj != 0:
            toffoli_std(qc, ctrl, X[j], sel)
            add_const_modN_rc_sel_inplace(qc, sel, Y, Kj, N, sel_one, carry, flag, ripple_direction=ripple_direction)
            toffoli_std(qc, ctrl, X[j], sel)
    ainv = pow(a, -1, N)
    for jj in range(n):
        j = _bit_index(jj, n, ripple_direction)
        weight = pow(2, n - 1 - j, N)
        Kj = (-ainv * weight) % N
        if Kj != 0:
            toffoli_std(qc, ctrl, Y[j], sel)
            add_const_modN_rc_sel_inplace(qc, sel, X, Kj, N, sel_one, carry, flag, ripple_direction=ripple_direction)
            toffoli_std(qc, ctrl, Y[j], sel)
    for i in range(n):
        cswap(qc, ctrl, X[i], Y[i])


# IQFT

def cphase(qc: "QuantumCircuit", theta, ctrl, tgt):
    qc.rz(theta / 2, tgt); qc.cx(ctrl, tgt); qc.rz(-theta / 2, tgt); qc.cx(ctrl, tgt)

def iqft_phase_general(qc: "QuantumCircuit", phase_regs, k_of_i: List[int], *, mode: str = "approx", cutoff: Optional[int] = 6) -> None:
    assert all(k_of_i[i] < k_of_i[i + 1] for i in range(len(k_of_i) - 1)), "k_of_i must be ascending"
    n = len(phase_regs)
    for i in reversed(range(n)):
        ki = k_of_i[i]
        for j in reversed(range(i + 1, n)):
            kj = k_of_i[j]
            gap = kj - ki
            if gap <= 0:
                continue
            if mode == "approx" and cutoff and gap > cutoff:
                continue
            cphase(qc, -math.pi / (2 ** gap), phase_regs[j], phase_regs[i])
        qc.h(phase_regs[i])
    for i in range(n // 2):
        qc.swap(phase_regs[i], phase_regs[n - 1 - i])


# Circuit builder

def build_shor_order_finder_parallel(
    N: int, a: int, ebits: int, *, iqft_mode: str = "full", iqft_cutoff: Optional[int] = None,
    toffoli_mode: str = "relphase", perm_id: int = 0, ripple_direction: str = "msb_to_lsb",
    reduction_strategy: str = "none", obfuscate: bool = False, rng_seed: int = 0,
    k_list: Optional[List[int]] = None, k_range: Optional[Tuple[int, int]] = None,
    pepper_logical: Optional[List[int]] = None, measure_junk: bool = False, compile_guard: bool = True,
    phase_mapping: Optional[Dict[str, object]] = None
) -> "QuantumCircuit":
    """Build a clean parallel order-finder with compacted window."""
    if k_range is not None and k_list is not None:
        raise ValueError("use either k_range or k_list")
    t = ebits
    if k_list is not None:
        k_of_i = sorted(set(int(k) for k in k_list))
        if min(k_of_i) < 0 or max(k_of_i) >= t:
            raise ValueError("k_list out of range")
    elif k_range is not None:
        lo, hi = k_range
        if not (0 <= lo <= hi < t):
            raise ValueError("k_range invalid")
        k_of_i = list(range(lo, hi + 1))
    else:
        k_of_i = list(range(t))

    phase = QuantumRegister(t, "ph")
    n = N.bit_length()
    xreg = QuantumRegister(n, "xr")
    yreg = QuantumRegister(n, "yr")
    anc = QuantumRegister(4, "anc")
    cbits = ClassicalRegister(t, "pc")
    qc = QuantumCircuit(phase, xreg, yreg, anc, cbits) if not measure_junk else QuantumCircuit(phase, xreg, yreg, anc, cbits, ClassicalRegister(4, "junk"))

    # compact mapping
    mapping = phase_mapping or build_phase_mapping(t, k_of_i)
    log2phys = mapping["log2phys"]

    # prepare |x>=|1>, superpose selected window
    qc.x(xreg[n - 1])
    for k in k_of_i:
        qc.h(phase[log2phys[k]])

    sel_one = anc[3]
    qc.x(sel_one)

    # controlled multiply-by-a for each window bit
    aconsts = {k: pow(a, 1 << k, N) for k in k_of_i}
    for idx, k in enumerate(k_of_i):
        ctrl = phase[log2phys[k]]
        modmul_const_rc_controlled_true_inplace(
            qc, ctrl=ctrl, X=xreg, Y=yreg, sel=anc[2], sel_one=sel_one,
            carry=anc[0], flag=anc[1], a=aconsts[k], N=N, ripple_direction="msb_to_lsb"
        )
        if compile_guard and ((idx + 1) % (1 if len(k_of_i) >= 10 else 2) == 0):
            qc.barrier(phase); qc.barrier(xreg); qc.barrier(yreg); qc.barrier(anc)

    # IQFT on the active window (respect logical gaps)
    phase_regs_window = [phase[log2phys[k]] for k in k_of_i]
    iqft_phase_general(qc, phase_regs_window, k_of_i, mode=iqft_mode, cutoff=iqft_cutoff)

    # measure
    if measure_junk:
        junk = qc.cregs[-1]
        for j in range(4):
            qc.measure(anc[j], junk[j])
    for k in k_of_i:
        qc.measure(phase[log2phys[k]], cbits[k])

    return qc


# Level profiles

LEVELS_PARALLEL = {
    "L0": {"n": 10, "t": 12}, "L1": {"n": 11, "t": 14}, "L2": {"n": 13, "t": 16}, "L3": {"n": 16, "t": 20},
    "L4": {"n": 19, "t": 24}, "L5": {"n": 22, "t": 28}, "L6": {"n": 25, "t": 32}, "L7": {"n": 28, "t": 36},
    "L8": {"n": 32, "t": 40}, "L9": {"n": 36, "t": 44},
    "D0": {"n": 6, "t": 6}, "D1": {"n": 8, "t": 8}, "D2": {"n": 10, "t": 10},
}

# Legacy guards for L0..L9
LEVEL_N_BAND = {
    "L0": (10, 11), "L1": (11, 12), "L2": (13, 14), "L3": (16, 17), "L4": (19, 20),
    "L5": (22, 23), "L6": (25, 26), "L7": (28, 29), "L8": (32, 33), "L9": (36, 37),
    "D0": (6, 6), "D1": (8, 8), "D2": (10, 10),
}

K_CAP = {
    "L0": 10, "L1": 14, "L2": 15, "L3": 18, "L4": 20, "L5": 22, "L6": 24, "L7": 28, "L8": 32, "L9": 36,
    "D0": 8, "D1": 10, "D2": 12,
}

ODD_MIN_BITS = {
    "L0": 1, "L1": 2, "L2": 3, "L3": 3, "L4": 4,
    "L5": 4, "L6": 5, "L7": 5, "L8": 6, "L9": 6,
    "D0": 1, "D1": 1, "D2": 2,
}

V2_RANGE = {
    "L0": (1, 6), "L1": (1, 5), "L2": (1, 5), "L3": (1, 6),
    "L4": (1, 7), "L5": (1, 8), "L6": (1, 9), "L7": (1, 10),
    "L8": (1, 11), "L9": (1, 12),
    "D0": (1, 4), "D1": (1, 4), "D2": (1, 5),
}

def parse_level(level: str) -> Tuple[str, Optional[int]]:
    level = level.strip().upper()
    if level.startswith("L") and level[1:].isdigit():
        return "L", int(level[1:])
    if level in ("D0", "D1", "D2"):
        return "D", None
    raise SystemExit(f"Unsupported level '{level}'")

def rule_based_profile(k: int) -> Dict[str, int]:
    """
    For L10+:
      - grow t slowly (+2/level) and n modestly (+1/level) to keep Q in check,
      - default guards in line with growth.
    Base from L9: t=44, n=36.
    """
    dk = max(0, k - 9)
    t = 44 + 2 * dk
    n = 36 + 1 * dk
    return {"n": n, "t": t}

def dynamic_kcap(t: int) -> int:
    return max(1, t - max(4, t // 6))

def dynamic_v2_range(t: int) -> Tuple[int, int]:
    v2_lo = 1
    v2_hi = max(6, min(12, t // 4))
    return (v2_lo, v2_hi)

def dynamic_odd_min_bits(t: int, base: int = 6) -> int:
    return max(base, min(12, (t // 4)))

def r_cap_bits_from(t: int, n: int) -> int:
    bits = min(int(math.floor(t / 2 - 0.5)), n - 2)
    return max(bits, 1)

def sample_r_in_band(t: int, n: int, alpha: float, *, rng: random.Random) -> int:
    r_bits = r_cap_bits_from(t, n)
    r_cap = 1 << r_bits
    lo = ((int(math.ceil(alpha * r_cap)) + 1) // 2) * 2
    hi = (r_cap // 2) * 2
    if lo > hi:
        lo = ((max(2, hi - 2) + 1) // 2) * 2
    even_count = ((hi - lo) // 2) + 1
    k = rng.randrange(even_count)
    return lo + 2 * k

def _sample_r_with_guards_dynamic(level_str: str, t: int, n_target: int, *, rng: random.Random, alpha: float,
                                  v2_lo: int, v2_hi: int, odd_min_bits: int) -> int:
    for _ in range(200):
        r = sample_r_in_band(t, n_target, alpha, rng=rng)
        s = v2(r)
        if not (v2_lo <= s <= v2_hi):
            continue
        m = r >> s
        if m <= 0 or m.bit_length() < odd_min_bits:
            continue
        return r
    return sample_r_in_band(t, n_target, alpha, rng=rng)


# Synthesis wrappers

def _synthesize_exact_order_and_n_band(
    r: int, *, n_target: int, n_band: Tuple[int, int], rng_seed: int, max_tries: int = 80, bit_slack: int = 1
) -> Tuple[int, int, int, int, int, int]:
    lo_n, hi_n = n_band
    tries = 0
    rng = random.Random(rng_seed)
    offsets = [0] + [d for k in range(1, max(1, bit_slack) + 1) for d in (+k, -k)]
    while tries < max_tries:
        for off in offsets:
            tgt = max(2, n_target + off)
            try:
                p, q, N, a, rp, rq = synthesize_semiprime_and_base_for_period(r, seed=rng.randrange(1 << 30), target_n_bits=tgt)
            except Exception:
                tries += 1
                continue
            n = N.bit_length()
            if not (lo_n <= n <= hi_n):
                tries += 1
                continue
            if pow(a, r, N) != 1:
                tries += 1
                continue
            ok = True
            for tprime in set(prime_factors(r).keys()):
                if pow(a, r // tprime, N) == 1:
                    ok = False
                    break
            if not ok:
                tries += 1
                continue
            return p, q, N, a, rp, rq
        tries += 1
    raise RuntimeError(f"Could not synthesize exact order r={r} with n in band {n_band} after {max_tries} attempts")

def _synth_with_r_resamples(
    r: int, t: int, n_target: int, n_band: Tuple[int, int], rng_seed: int, *,
    max_r_retries: int = 12, max_tries: int = 100, bit_slack: int = 2, alpha: float = 0.70
) -> Tuple[int, int, int, int, int, int, int]:
    for _ in range(max_r_retries):
        try:
            p, q, N, a, rp, rq = _synthesize_exact_order_and_n_band(r, n_target=n_target, n_band=n_band, rng_seed=rng_seed, max_tries=max_tries, bit_slack=bit_slack)
            return p, q, N, a, rp, rq, r
        except RuntimeError:
            r = sample_r_in_band(t, n_target, alpha, rng=random)
    raise SystemExit(f"Failed to find (N,a) with exact order and n-band after {max_r_retries} r resamples")


# Protocol emission

def _emit_protocol_variants(
    *, outdir: Path, level: str, t: int, N: int, a: int, r: int, n_bits: int,
    k_list_canonical: List[int],
    beacon: str, miner_id: str, variants: int,
    recommended_shots: int, deadline_minutes: int, rng_seed: int
) -> None:
    """
    Emit K salted QASM variants + meta.json (public) + meta_private.json (private).
    Sliding contiguous windows and compact mapping.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    challenge_id = f"{time.strftime('%Y%m%d')}-{level}-{miner_id}"

    deadline_dt = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=int(deadline_minutes))).replace(microsecond=0)
    deadline_utc = deadline_dt.isoformat().replace("+00:00", "Z")

    base_width = len(k_list_canonical)

    variants_list: List[Dict[str, object]] = []

    for i in range(int(variants)):
        salt = derive_salt(beacon, miner_id, i)
        ks_i = klist_variant_from_salt(salt, t, base_len=base_width, idx=i)  # salted for v0 too

        # Always full IQFT
        iqft_i = "full"
        cut_i = None

        mapping = build_phase_mapping(t, ks_i)

        qc_i = build_shor_order_finder_parallel(
            N, a, t,
            iqft_mode=iqft_i,
            iqft_cutoff=cut_i,
            toffoli_mode="relphase",
            perm_id=0,
            ripple_direction="msb_to_lsb",
            reduction_strategy="none",
            obfuscate=False,
            rng_seed=(rng_seed ^ int(salt[:16], 16)),
            k_list=ks_i,
            pepper_logical=None,
            measure_junk=False,
            compile_guard=True,
            phase_mapping=mapping,
        )

        file_i = f"{level}_v{i}.qasm"
        out_path = outdir / file_i
        _atomic_write_qasm(qc_i, out_path)
        q_sha = qasm_sha256_file(out_path)

        variants_list.append({
            "index": i,
            "file": file_i,
            "qasm_sha256": q_sha,
            "salt": salt,
            "k_list": ks_i,
            "iqft_mode": iqft_i,
            "iqft_cutoff": None,
            "phase_logical_to_physical": mapping["log2phys"],
        })

    rng_local = random.Random(rng_seed ^ 0x9E3779B97F4A7C15)
    nonce_delta = rng_local.getrandbits(t)

    # Public meta
    public_meta = {
        "schema": "qr-shor-challenge/v1",
        "challenge_id": challenge_id,
        "level": level,
        "beacon": beacon,
        "miner_id": miner_id,
        "t": t,
        "recommended_shots": int(recommended_shots),
        "deadline_utc": deadline_utc,
        "rng_seed": int(rng_seed),
        "variants": variants_list,
    }
    _atomic_write_text(outdir / "meta.json", json.dumps(public_meta, indent=2))

    # Private meta with δ
    private_meta = {
        **public_meta,
        "secret": {
            "N": int(N), "a": int(a), "r": int(r),
            "selection": {"k_list_canonical": k_list_canonical},
            "n": int(n_bits),
            "rng_seed": int(rng_seed),
            "nonce_delta": int(nonce_delta),
            "nonce_scheme": "verifier_offset_v1",
        },
    }

    private_meta = _attach_verifier_bindings(private_meta)

    _atomic_write_text(outdir / "meta_private.json", json.dumps(private_meta, indent=2))
    print(f"[pro] Wrote {len(variants_list)} variants and meta.json to {outdir}")


# CLI

def main():
    ap = argparse.ArgumentParser(description="Shor QASM generator (parallel mode; single or protocol variants)")
    ap.add_argument("--level", required=True, help="Lk or Dk (L10+ supported)")
    ap.add_argument("--rng-seed", type=int, default=None)
    ap.add_argument("--emit-json", action="store_true")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--k-list", default=None)
    ap.add_argument("--k-range", default=None)

    # protocol flags
    ap.add_argument("--beacon", default=None, help="hex randomness beacon (public)")
    ap.add_argument("--miner-id", default=None, help="miner identifier (public)")
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--recommended-shots", type=int, default=384)
    ap.add_argument("--deadline-minutes", type=int, default=90)

    args = ap.parse_args()

    # Manual k selection
    k_list = None
    k_range = None
    if args.k_list and args.k_range:
        raise SystemExit("Use either --k-list or --k-range (not both).")
    if args.k_list:
        try:
            k_list = [int(s) for s in args.k_list.split(",") if s.strip()]
        except Exception:
            raise SystemExit("--k-list must be 'k1,k2,...'")
    if args.k_range:
        try:
            lo_s, hi_s = args.k_range.split(":")
            lo, hi = int(lo_s), int(hi_s)
            if lo < 0 or hi < lo:
                raise ValueError
            k_range = (lo, hi)
        except Exception:
            raise SystemExit("--k-range must be 'lo:hi'")

    # Seed once
    rng_seed = _seed_from_protocol(args.beacon, args.miner_id, args.rng_seed)
    random.seed(rng_seed)
    rng = random

    # Resolve level profile
    kind, kval = parse_level(args.level)
    if kind == "D":
        cfg = LEVELS_PARALLEL[args.level]
        n_target, t = cfg["n"], cfg["t"]
        lo_n, hi_n = (n_target, n_target)
        k_cap = K_CAP.get(args.level, t)
        v2_lo, v2_hi = V2_RANGE[args.level]
        odd_min_bits = ODD_MIN_BITS[args.level]
        window_ratio = 0.8
    else:
        # L-level
        label = f"L{kval}"
        if label in LEVELS_PARALLEL:
            # legacy exact behavior for L0..L9
            cfg = LEVELS_PARALLEL[label]
            n_target, t = cfg["n"], cfg["t"]
            lo_n, hi_n = LEVEL_N_BAND.get(label, (n_target, n_target))
            k_cap = K_CAP.get(label, t)
            v2_lo, v2_hi = V2_RANGE.get(label, (1, max(6, min(12, t // 4))))
            odd_min_bits = ODD_MIN_BITS.get(label, max(4, t // 4))
            # Legacy window ratio mild
            window_ratio = 0.55 if kval <= 2 else 0.65
        else:
            # rule-based for L10+
            prof = rule_based_profile(kval)
            n_target, t = prof["n"], prof["t"]
            lo_n, hi_n = (n_target, n_target)  # stay near target bit-length
            k_cap = dynamic_kcap(t)
            v2_lo, v2_hi = dynamic_v2_range(t)
            odd_min_bits = dynamic_odd_min_bits(t)
            window_ratio = min(0.9, 0.70 + 0.02 * (kval - 5))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    alpha = 0.70 if kind == "D" or (kind == "L" and (kval or 0) <= 5) else min(0.84, 0.70 + 0.02 * (kval - 5))
    r = _sample_r_with_guards_dynamic(args.level, t, n_target, rng=rng, alpha=alpha, v2_lo=v2_lo, v2_hi=v2_hi, odd_min_bits=odd_min_bits)
    p, q, N, a, rp, rq, r = _synth_with_r_resamples(r, t, n_target, (lo_n, hi_n), rng_seed, max_r_retries=12, max_tries=100, bit_slack=2, alpha=alpha)
    n_bits = N.bit_length()

    # Determine k_list (manual overrides win). If not provided, RANDOMIZE the window.
    if k_list is not None or k_range is not None:
        # Manual path wins
        if k_list is None and k_range is not None:
            ks = list(range(k_range[0], k_range[1] + 1))
        else:
            ks = k_list
        mapping = build_phase_mapping(t, ks)

        # Always full IQFT
        iqft_mode_local = "full"
        iqft_cut_local = None

        qc = build_shor_order_finder_parallel(
            N, a, t,
            iqft_mode=iqft_mode_local, iqft_cutoff=iqft_cut_local,
            toffoli_mode="relphase", perm_id=0, ripple_direction="msb_to_lsb", reduction_strategy="none",
            obfuscate=False, rng_seed=rng_seed, k_list=ks, pepper_logical=None, measure_junk=False, compile_guard=True, phase_mapping=mapping
        )
    else:
        w_target = max(2, min(int(round(window_ratio * t)), int(k_cap)))
        s = v2(r)
        m_odd = r >> s
        w_odd = int(math.ceil(math.log2(m_odd))) + 1 if m_odd > 1 else 2
        w_target = max(w_target, w_odd)
        max_shift = max(0, t - w_target)
        start = rng.randrange(0, max_shift + 1)
        ks = list(range(start, start + w_target))

        # Always full IQFT
        iqft_mode_local = "full"
        iqft_cut_local = None

        qc = build_shor_order_finder_parallel(
            N, a, t,
            iqft_mode=iqft_mode_local, iqft_cutoff=iqft_cut_local,
            toffoli_mode="relphase", perm_id=0, ripple_direction="msb_to_lsb", reduction_strategy="none",
            obfuscate=False, rng_seed=rng_seed, k_list=ks, pepper_logical=None, measure_junk=False, compile_guard=True,
            phase_mapping=build_phase_mapping(t, ks)
        )

    # Report Q
    Q = t + 2 * n_bits + 4

    if args.variants >= 1:
        beacon_use = (args.beacon or secrets.token_hex(32))
        miner_use  = (args.miner_id or "anon")
        rec_shots = int(args.recommended_shots if args.recommended_shots else max(256, 24 * len(ks)))
        _emit_protocol_variants(
            outdir=outdir, level=args.level, t=t, N=N, a=a, r=r, n_bits=n_bits,
            k_list_canonical=ks,
            beacon=beacon_use, miner_id=miner_use, variants=args.variants,
            recommended_shots=rec_shots, deadline_minutes=args.deadline_minutes, rng_seed=rng_seed
        )
        if args.emit_json:
            sol = {
                "level": args.level, "mode": "parallel", "n": n_bits, "t": t, "Q": Q, "N": N, "a": a, "r": r,
                "selection": {"k_list": ks}, "n_band": {"lo": lo_n, "hi": hi_n}, "rng_seed": rng_seed
            }
            _atomic_write_text(outdir / "shor_out.json", json.dumps(sol, indent=2))
        return

    # (Fallback) Single QASM output (used if --variants 0)
    fixed_qasm = outdir / "shor_out.qasm"
    _atomic_write_qasm(qc, fixed_qasm)
    if args.emit_json:
        sol = {
            "level": args.level, "mode": "parallel", "n": n_bits, "t": t, "Q": Q, "N": N, "a": a, "r": r,
            "selection": {"k_list": ks}, "n_band": {"lo": lo_n, "hi": hi_n}, "rng_seed": rng_seed
        }
        _atomic_write_text(outdir / "shor_out.json", json.dumps(sol, indent=2))
    print(f"[{args.level}] n={n_bits} t={t} Q={Q} r={r} |k|={len(ks)}")
    print(f"[ok] Wrote QASM: {fixed_qasm}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[error] Unhandled exception:", e)
        traceback.print_exc()
        raise
