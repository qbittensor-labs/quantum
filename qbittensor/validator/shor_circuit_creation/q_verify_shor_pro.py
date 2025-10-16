#!/usr/bin/env python3
"""
q_verify_shor_pro.py — strict single-circuit verifier
"""

from __future__ import annotations
import argparse, json, math, os, sys
from typing import List, Tuple, Optional


def die(msg: str, code: int = 1) -> None:
    print(f"[fatal] {msg}", file=sys.stderr, flush=True); sys.exit(code)

def read_json(path: str) -> dict:
    if not os.path.exists(path): die(f"File not found: {path}")
    with open(path, "r") as f: return json.load(f)

def read_counts_txt(path: str) -> List[Tuple[int, int]]:
    """TXT format: '<bitstring> <count>' (MSB-left)."""
    items: List[Tuple[int,int]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            bits, *rest = line.split()
            cnt = int(rest[0]) if rest else 1
            items.append((int(bits, 2), cnt))
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items

def spacing_bins(W: int, r: int) -> float:
    U = min(int(r), 1 << int(W))
    return (1 << int(W)) / float(U if U > 0 else 1)

def choose_radius(Delta: float, shots: int, Bw_user: int|None, *, coverage_target: float = 0.30) -> int:
    """
    Auto-choose Bw so c ≈ coverage_target
    """
    limit = max(0, int((int(Delta)//2) - 1))
    if Bw_user is not None:
        return max(0, min(limit, int(Bw_user)))
    bw = int(math.floor(((coverage_target * float(Delta)) - 1.0) / 2.0))
    return max(0, min(limit, bw))

def pack_bits_from_positions(x: int, positions: List[int]) -> int:
    v = 0
    for i, p in enumerate(positions):
        v |= ((x >> p) & 1) << i
    return v

def nearest_center_int(sW: int, W: int, r: int) -> int:
    two_W = 1 << W
    k = (sW * r + (two_W // 2)) // two_W
    c = (k * two_W + r // 2) // r
    return int(c)

def score_counts_windowed_int(count_items: List[Tuple[int,int]], *,
                              positions: List[int], delta_w: int, W: int, r: int,
                              Bw: int, thr: float, eps: float,
                              early_exit: bool = True) -> Tuple[float,int]:
    mask = (1 << W) - 1
    total = sum(c for _, c in count_items)
    good = 0; seen = 0
    for s_raw, c in count_items:
        sW = pack_bits_from_positions(s_raw, positions)
        sW = (sW + delta_w) & mask  # apply +δ in window basis
        center = nearest_center_int(sW, W, r)
        d = (sW - center) & mask
        d = min(d, (mask + 1) - d)
        if d <= Bw: good += c
        seen += c
        if early_exit:
            potential = (good + (total - seen)) / max(1, total)
            if potential + 1e-12 < (thr - eps): break
    return (good / max(1, total), total)

def baseline_p0_windowed_int(W: int, r: int, Bw: int) -> float:
    Delta = spacing_bins(W, r)
    return min(1.0, (2*Bw + 1) / max(1.0, Delta))

def choose_threshold_doc(p0: float, shots: int, *, min_thr=0.15, max_thr=0.35) -> Tuple[float, float]:
    """
    thr = clamp(3*p0, min_thr, max_thr)
    eps = max(0.005, C_eps * sqrt(p(1-p)/shots)), with a touch more slack at modest shots.
    """
    p = max(0.0, min(1.0, p0))
    sigma = math.sqrt(p * max(1.0 - p, 1e-12) / max(1, shots))
    thr = max(min_thr, min(max_thr, 3.0 * p))
    C_eps = 2.0 if shots <= 4096 else 1.5
    eps = max(0.005, C_eps * sigma)
    return thr, eps

def residue_alignment_diag(count_items: List[Tuple[int,int]], *,
                           positions: List[int], delta_w: int, W: int, r: int, Bw: int) -> Tuple[int,int,int,int,float]:
    """Diagnostic: build H, sliding-window masses, return (Di, mass0, best_shift, massbest, Delta)."""
    mask = (1 << W) - 1
    Delta = spacing_bins(W, r); Di = max(1, int(round(Delta)))
    H = [0]*Di
    for s_raw, c in count_items:
        sW = pack_bits_from_positions(s_raw, positions)
        sW = (sW + delta_w) & mask
        H[sW % Di] += c
    masses = [0]*Di
    tot = 0
    for j in range(-Bw, Bw+1):
        tot += H[(0 + j) % Di]
    masses[0] = tot
    for s in range(1, Di):
        add_idx = (s + Bw) % Di
        rem_idx = (s - 1 - Bw) % Di
        tot += H[add_idx]; tot -= H[rem_idx]
        masses[s] = tot
    best_shift = max(range(Di), key=lambda s: masses[s])
    mass_best = masses[best_shift]
    mass_zero = masses[0]
    return Di, mass_zero, best_shift, mass_best, float(Delta)

def _parse_level_tag(meta: dict) -> str:
    tag = str(meta.get("level") or "")
    if tag and tag[0].upper() == "L" and tag[1:].isdigit():
        return tag.upper()
    cid = str(meta.get("challenge_id") or "")
    for tok in cid.split("-"):
        if tok and tok[0].upper() == "L" and tok[1:].isdigit():
            return tok.upper()
    return ""

def _level_presets(lvl_num: int) -> dict:
    """
    Keys:
      Bw_fixed: int|None (force Bw or let chooser use 'cov')
      cov: float|None (target coverage for Bw auto-chooser)
      cov_cap: float|None (max allowed coverage (2*Bw+1)/Delta)
      Kp: float (peakness sigma)
      Ku: float (window-uplift sigma)
      cap: float (max threshold clamp)
      tau: float (delta-alignment tolerance as fraction of shots)
      div: float (diversity fraction within ±Bw window)
    """
    PRE = {
        0: {"Bw_fixed": 1,    "cov": 0.70, "cov_cap": 0.80,
            "Kp": 1.90, "Ku": 1.80, "cap": 0.31,  "tau": 0.30,  "div": 0.30},
        1: {"Bw_fixed": 0,    "cov": None, "cov_cap": None,
            "Kp": 2.7,  "Ku": 2.3,  "cap": 0.35,  "tau": 0.015, "div": 0.30},
        2: {"Bw_fixed": 1,    "cov": 0.66, "cov_cap": 0.75,
            "Kp": 1.90, "Ku": 1.70, "cap": 0.285, "tau": 0.060, "div": 0.30},
        3: {"Bw_fixed": 0,    "cov": None, "cov_cap": None,
            "Kp": 2.5,  "Ku": 2.2,  "cap": 0.33,  "tau": 0.012, "div": 0.30},
        4: {"Bw_fixed": None, "cov": 0.35, "cov_cap": 0.38,
            "Kp": 2.15, "Ku": 2.25, "cap": 0.310, "tau": 0.025, "div": 0.25},
        5: {"Bw_fixed": None, "cov": 0.35, "cov_cap": 0.38,
            "Kp": 2.00, "Ku": 2.00, "cap": 0.310, "tau": 0.020, "div": 0.25},
        6: {"Bw_fixed": None, "cov": 0.30, "cov_cap": 0.33,
            "Kp": 2.2,  "Ku": 2.0,  "cap": 0.35,  "tau": 0.010, "div": 0.30},
        7: {"Bw_fixed": None, "cov": 0.30, "cov_cap": 0.33,
            "Kp": 2.2,  "Ku": 2.0,  "cap": 0.35,  "tau": 0.010, "div": 0.30},
        8: {"Bw_fixed": None, "cov": 0.30, "cov_cap": 0.33,
            "Kp": 2.2,  "Ku": 2.0,  "cap": 0.35,  "tau": 0.008, "div": 0.30},
        9: {"Bw_fixed": None, "cov": 0.30, "cov_cap": 0.33,
            "Kp": 2.2,  "Ku": 2.0,  "cap": 0.35,  "tau": 0.008, "div": 0.30},
    }
    return PRE.get(lvl_num, PRE[6])



# Sliding caches
H_cache: Optional[List[int]] = None
masses_cache: Optional[List[int]] = None

def ensure_H_and_masses(items, positions, delta_w, W, Di, Bw):
    """Build H_cache and masses_cache in O(Di)."""
    global H_cache, masses_cache
    if H_cache is None:
        mask = (1 << W) - 1
        H_cache = [0] * Di
        for s_raw, c in items:
            sW = pack_bits_from_positions(s_raw, positions)
            sW = (sW + delta_w) & mask
            H_cache[sW % Di] += c
    if masses_cache is None:
        masses_cache = [0] * Di
        tot = 0
        for j in range(-Bw, Bw + 1):
            tot += H_cache[(0 + j) % Di]
        masses_cache[0] = tot
        for s in range(1, Di):
            add_idx = (s + Bw) % Di
            rem_idx = (s - 1 - Bw) % Di
            tot += H_cache[add_idx]
            tot -= H_cache[rem_idx]
            masses_cache[s] = tot

def run_once(meta: dict, items: List[Tuple[int,int]], *,
             positions: List[int],
             delta_w: int,
             W: int,
             r: int,
             t: int,
             args: argparse.Namespace,
             preset: dict,
             enc_label: str,
             s_anchor_override: Optional[int] = None) -> Tuple[bool, str, float]:
    """
    Executes one 'encoding' attempt. Returns (ok, message, margin_or_negative_score).

    Positive margin => pass; more negative means a 'stronger' fail.
    """
    global H_cache, masses_cache
    # Reset caches each try (encoding changes mapping)
    H_cache = None; masses_cache = None

    # Per-level presets
    cov_tgt = args.coverage_target
    cov_cap = args.max_coverage
    Bw_fixed = args.Bw

    if args.coverage_target == 0.30 and preset.get("cov") is not None:
        cov_tgt = preset["cov"]
    if args.max_coverage == 0.33 and preset.get("cov_cap") is not None:
        cov_cap = preset["cov_cap"]
    if args.Bw is None and preset.get("Bw_fixed") is not None:
        Bw_fixed = preset["Bw_fixed"]

    K_peak_eff  = float(args.peak_guard_sigma) if (args.peak_guard_sigma  is not None) else preset["Kp"]
    Kw_eff      = float(args.uplift_guard_sigma) if (args.uplift_guard_sigma is not None) else preset["Ku"]
    max_thr_cap = float(args.max_thr_cap)       if (args.max_thr_cap      is not None) else preset["cap"]
    tau_frac    = float(args.delta_align_tau_frac) if (args.delta_align_tau_frac is not None) else preset["tau"]
    div_frac    = float(args.diversity_frac)    if (args.diversity_frac   is not None) else preset["div"]

    # Geometry & radius
    Delta = spacing_bins(W, r)
    total_shots = sum(c for _,c in items)
    Bw = choose_radius(Delta, total_shots, Bw_user=Bw_fixed, coverage_target=cov_tgt)

    # Coverage cap
    coverage = (2*Bw + 1) / max(1e-9, float(Delta))
    if coverage > float(cov_cap) + 1e-12:
        msg = (f"[enc] {enc_label}\n"
               f"[FAIL] coverage too high: (2*Bw+1)/Delta={coverage:.3f} with Bw={Bw}, Delta≈{Delta:.2f}. "
               f"Lower Bw or widen W (regenerate).")
        return (False, msg, -1.0)

    # Diagnostics residues
    Di, m0_res, bshift_diag, mbest_res, Delta_f = residue_alignment_diag(
        items, positions=positions, delta_w=delta_w, W=W, r=r, Bw=Bw
    )

    # Dynamic softening for fragile geometry (pre-guard)
    if Di <= 5 and Bw == 0:
        tau_frac   = max(tau_frac, 0.06)
        Kw_eff     = min(Kw_eff, 1.8)
        K_peak_eff = min(K_peak_eff, 1.9)

    # Threshold & eps
    p0 = baseline_p0_windowed_int(W, r, Bw)
    thr, eps = choose_threshold_doc(p0, total_shots, min_thr=0.15, max_thr=max_thr_cap)

    # Report header for this encoding
    head = (f"[enc] {enc_label}\n"
            f"[build] guarded verifier: coverage_target={cov_tgt:.2f}, max_coverage={cov_cap:.2f}, "
            f"res_align={args.res_align}, peak_guard_sigma={K_peak_eff:.2f}, uplift_sigma={Kw_eff:.2f}, "
            f"max_thr_cap={max_thr_cap:.3f}, tau_frac={tau_frac:.4f}, diversity_frac={div_frac:.2f}\n"
            f"[build] spike_guard=bitstring\n"
            f"[thr] WININT[strict][Bw={Bw}]: W={W} Delta≈{Delta:.2f} p0={p0:.4f} "
            f"shots={total_shots} → thr={thr:.3f} eps={eps:.4f}\n"
            f"[diag] Delta≈{Delta_f:.2f} Di={Di} Bw={Bw} delta_w={delta_w} "
            f"mass(0)={m0_res} best_shift={bshift_diag} mass(best)={mbest_res} second≈(see code)")

    # Reset caches (fresh for scoring)
    H_cache = None; masses_cache = None

    # Window mass and optional residue alignment uplift
    mass0, total = score_counts_windowed_int(
        items, positions=positions, delta_w=delta_w, W=W, r=r, Bw=Bw, thr=thr, eps=eps,
        early_exit=(not args.no_early_exit)
    )
    mass_used = float(mass0)
    if args.res_align != "off" and Di > 1:
        ensure_H_and_masses(items, positions, delta_w, W, Di, Bw)
        if args.res_align == "micro":
            shifts = [0, 1, (Di - 1)]
            mb = max(masses_cache[s] for s in shifts)
            mass_used = max(mass_used, mb / max(1, total))
        else:
            mass_used = max(mass_used, float(max(masses_cache)) / max(1, total))

    # Residue-based guards
    if Di > 1:
        ensure_H_and_masses(items, positions, delta_w, W, Di, Bw)

        # Flatness + peakness
        E = max(1.0, float(total) / float(Di))
        chisq = 0.0
        for hi in H_cache:
            diff = float(hi) - E
            chisq += (diff * diff) / E
        df = max(1, Di - 1)
        chisq_cut = df + 2.0 * math.sqrt(2.0 * df)

        base_res  = 1.0 / float(Di)
        sigma_res = math.sqrt(base_res * max(1.0 - base_res, 1e-12) / max(1, total))
        top_center = max(H_cache) / max(1, total)

        not_peaky = (top_center + 1e-12 < (base_res + K_peak_eff * sigma_res))
        if not_peaky and (chisq + 1e-12 < chisq_cut):
            msg = (f"{head}\n"
                   f"[FAIL] flatness guard: not-peaky and chi²-flat "
                   f"(top={top_center:.3f} < base+{K_peak_eff:.1f}σ={(base_res + K_peak_eff*sigma_res):.3f}, "
                   f"chi²={chisq:.1f} < cut≈{chisq_cut:.1f}; Di={Di}, shots={total}).")
            return (False, msg, top_center - (base_res + K_peak_eff * sigma_res))

        # Spike (bitstring)
        max_single = max(c for _, c in items) / max(1, total)
        if max_single > 0.60:
            msg = (f"{head}\n"
                   f"[diag] spike_probe: max_single={max_single:.3f}\n"
                   f"[FAIL] spike guard (bitstring): top_bitstring_frac={max_single:.3f} > 0.60 (shots={total}).")
            return (False, msg, 0.60 - max_single)

        # Peakness (single residue)
        if top_center + 1e-12 < (base_res + K_peak_eff * sigma_res):
            msg = (f"{head}\n"
                   f"[diag] spike_probe: max_single={max_single:.3f}\n"
                   f"[FAIL] peakness guard: top_residue={top_center:.3f} "
                   f"< base+{K_peak_eff:.1f}σ={(base_res + K_peak_eff*sigma_res):.3f} "
                   f"(Di={Di}, shots={total}).")
            return (False, msg, top_center - (base_res + K_peak_eff * sigma_res))

        # C1: Anchor-aware δ-alignment tolerance
        mass_zero = masses_cache[0]
        mass_best = max(masses_cache)
        bshift = masses_cache.index(mass_best)
        s_anchor = int(meta.get("secret", {}).get("residue_anchor", 0)) % max(1, Di)
        if s_anchor_override is not None:
            s_anchor = int(s_anchor_override) % max(1, Di)
        shift_err = (bshift - s_anchor) % max(1, Di)
        tau = max(1.0, tau_frac * total)
        if shift_err != 0 and (mass_best - mass_zero) > tau:
            msg = (f"{head}\n"
                   f"[diag] spike_probe: max_single={max_single:.3f}\n"
                   f"[FAIL] delta-alignment guard: best_shift={bshift} (anchor={s_anchor}) "
                   f"→ shift_err={shift_err} by {(mass_best - mass_zero)} counts "
                   f"(> {tau:.1f} tol @ {tau_frac*100:.1f}% of shots). "
                   f"(Di={Di}, Bw={Bw}, shots={total})")
            return (False, msg, float(tau - (mass_best - mass_zero)))

        # Diversity: require spread only when window has meaningful width
        nbins = 2*Bw + 1
        if nbins >= 5:
            need = max(4, int(math.ceil(div_frac * nbins)))
            nonzero = 0
            for j in range(-Bw, Bw+1):
                if H_cache[((s_anchor + j) % Di)] > 0:
                    nonzero += 1
            if nonzero < need:
                msg = (f"{head}\n"
                       f"[diag] spike_probe: max_single={max_single:.3f}\n"
                       f"[FAIL] diversity guard: only {nonzero}/{nbins} residues non-empty in ±Bw window around anchor "
                       f"(need ≥ {need}; frac {div_frac:.2f}). (Di={Di}, Bw={Bw}, shots={total})")
                return (False, msg, float(nonzero - need))

    # Uplift
    sigma_w = math.sqrt(p0 * max(1.0 - p0, 1e-12) / max(1, total))
    uplift_cut = (p0 + Kw_eff * sigma_w)
    if (mass_used + 1e-12) < uplift_cut:
        msg = (f"{head}\n"
               f"[FAIL] window-uplift guard: mass={mass_used:.3f} < p0+{Kw_eff:.1f}σ={uplift_cut:.3f} "
               f"(p0={p0:.3f}, σ_w={sigma_w:.4f}, shots={total}).")
        return (False, msg, mass_used - uplift_cut)

    # Final clamp
    margin = (mass_used + eps) - thr
    ok = (margin >= 0.0)
    tag = "OK" if ok else "FAIL"
    final = (f"{head}\n"
             f"[{tag}] {meta['variants'][0]['file']}: mass={mass_used:.3f} (thr {thr:.3f}, margin {margin:.3f}, "
             f"cfg=strict|Bw={Bw}, r_used={r}, shots={total}) "
             f"[w={W}, t={t}]")
    return (ok, final, margin)

# Encoding-sweep wrapper

def run_once_with_anchor_mode(meta: dict, items: List[Tuple[int,int]], *,
                              positions: List[int],
                              delta_w: int,
                              W: int,
                              r: int,
                              t: int,
                              args: argparse.Namespace,
                              preset: dict,
                              enc_label: str,
                              s_anchor_mode: Optional[str] = None) -> Tuple[bool, str, float]:
    """
    Thin wrapper over run_once to optionally negate the anchor when δ sign is flipped.
    """
    # derive Di to transform anchor if requested
    Delta = spacing_bins(W, r); Di = max(1, int(round(Delta)))
    s_anchor = int(meta.get("secret", {}).get("residue_anchor", 0)) % max(1, Di)
    if s_anchor_mode == "negate_anchor" and Di > 1:
        s_anchor = (-s_anchor) % Di
    return run_once(meta, items, positions=positions, delta_w=delta_w, W=W, r=r, t=t,
                    args=args, preset=preset, enc_label=enc_label, s_anchor_override=s_anchor)

def main():
    ap = argparse.ArgumentParser(description="Strict Shor verifier (single circuit) with level presets.")
    ap.add_argument("meta_private", help="Path to meta_private.json")
    ap.add_argument("results_dir", help="Directory with <base>.counts.txt")
    ap.add_argument("--Bw", type=int, default=None, help="Fix integer radius; else auto from coverage target")
    ap.add_argument("--coverage-target", type=float, default=0.30)
    ap.add_argument("--max-coverage", type=float, default=0.33)
    ap.add_argument("--res-align", choices=["off","micro","best"], default="best")
    ap.add_argument("--peak-guard-sigma", type=float, default=None)
    ap.add_argument("--uplift-guard-sigma", type=float, default=None)
    ap.add_argument("--max-thr-cap", type=float, default=None)
    ap.add_argument("--delta-align-tau-frac", type=float, default=None)
    ap.add_argument("--diversity-frac", type=float, default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--no-early-exit", action="store_true")
    # Default: encoding sweep ON. Opt out with --no-auto-enc.
    ap.add_argument("--no-auto-enc", action="store_true",
                    help="Disable encoding sweep (bit-order asc/desc × δ sign +/-).")
    args = ap.parse_args()

    meta = read_json(args.meta_private)
    try:
        t = int(meta["t"])
        secret = meta["secret"]; r = int(secret["r"]); delta_t = int(secret["nonce_delta"])
        v = meta["variants"][0]
        base = os.path.splitext(v["file"])[0]
        k_list = list(v.get("k_list") or secret.get("selection", {}).get("k_list_canonical", []))
        if not k_list: die("No k_list found in meta.")
    except Exception as e:
        die(f"meta_private.json missing fields: {e}")

    counts_path = os.path.join(args.results_dir, f"{base}.counts.txt")
    if not os.path.exists(counts_path):
        die(f"Counts not found: {counts_path}")
    items = read_counts_txt(counts_path)

    positions_sorted = sorted(int(k) for k in k_list)
    W = len(positions_sorted)
    if W <= 0 or W > t: die(f"Bad window width W={W} for t={t}")

    level_tag = _parse_level_tag(meta)
    try:
        lvl_num = int(level_tag[1:]) if (level_tag and level_tag[0].upper()=="L") else 0
    except Exception:
        lvl_num = 0
    preset = _level_presets(lvl_num)

    # Helper to derive δ in window basis for a given bit-order and sign.
    def enc_delta(positions: List[int], sign: int) -> int:
        mask = (1 << W) - 1
        dw = pack_bits_from_positions(delta_t, positions) & mask
        if sign < 0:
            dw = (-dw) & mask
        return dw

    auto_enc = (not args.no_auto_enc)
    trials: List[Tuple[str, List[int], int, Optional[str]]] = []
    if auto_enc:
        asc = positions_sorted
        desc = positions_sorted[::-1]
        trials.append(("pos=asc,delta=+", asc,  +1, None))
        trials.append(("pos=asc,delta=-", asc,  -1, "negate_anchor"))
        trials.append(("pos=desc,delta=+", desc, +1, None))
        trials.append(("pos=desc,delta=-", desc, -1, "negate_anchor"))
    else:
        asc = positions_sorted
        trials.append(("pos=asc,delta=+", asc, +1, None))

    # Track the strongest failure for final reporting (most negative margin)
    best_fail_msg = None
    best_fail_score = +1e9  # choose the MINIMUM (most negative) margin
    for enc_label, pos, sign, anchor_mode in trials:
        dw = enc_delta(pos, sign)
        ok, msg, margin = run_once_with_anchor_mode(
            meta, items, positions=pos, delta_w=dw, W=W, r=r, t=t,
            args=args, preset=preset, enc_label=enc_label, s_anchor_mode=anchor_mode
        )
        # Always print each try for visibility
        print(msg, flush=True)
        if ok:
            sys.exit(0)
        if margin < best_fail_score:
            best_fail_score = margin
            best_fail_msg = msg

    # If here, all encodings failed — report strongest diagnostic
    print(best_fail_msg or "[FAIL] all encodings failed (no additional diagnostic).", flush=True)
    sys.exit(2)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print("[error] Unhandled exception:", e)
        traceback.print_exc()
        raise
