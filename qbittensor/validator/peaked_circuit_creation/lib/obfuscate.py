import os
from typing import Tuple, List
import numpy as np
import bittensor as bt

from .circuit import SU4
_KRON_ORDER = os.getenv('QBT_OBF_KRON_ORDER', '01').strip()

def _xor_bitstring(a: str, mask: np.ndarray) -> str:
    arr = np.frombuffer(a.encode('ascii'), dtype=np.uint8) - ord('0')
    arr ^= mask.astype(np.uint8)
    return ''.join('1' if x else '0' for x in arr.tolist())

_X2 = np.array([[0.0 + 0.0j, 1.0 + 0.0j],
                [1.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex128)
_I2 = np.eye(2, dtype=np.complex128)


def _pauli_x() -> np.ndarray:
    return _X2

def _identity() -> np.ndarray:
    return _I2

def _random_u2(rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal(size=(2, 2)) + 1j * rng.standard_normal(size=(2, 2))
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    ph = d / np.abs(d)
    u = q * ph
    return u.astype(np.complex128, copy=False)

def _left_right_kron(left: np.ndarray | None, right: np.ndarray | None) -> np.ndarray:
    l = left if left is not None else _identity()
    r = right if right is not None else _identity()
    if _KRON_ORDER == '10':
        return np.kron(r, l)
    return np.kron(l, r)

def _swap4() -> np.ndarray:
    s = np.zeros((4, 4), dtype=np.complex128)
    s[0, 0] = 1.0
    s[1, 2] = 1.0
    s[2, 1] = 1.0
    s[3, 3] = 1.0
    return s

def _build_occurrences(unis: List[SU4], num_qubits: int) -> List[List[int]]:
    occ: List[List[int]] = [[] for _ in range(num_qubits)]
    for idx, u in enumerate(unis):
        occ[u.target0].append(idx)
        occ[u.target1].append(idx)
    return occ

def obfuscate_su4_series(
    target_state: str,
    unis: List[SU4],
    seed: int,
    flip_rate_override: float | None = None,
    ensure_touch_all: bool = False,
    flip_rate_env: str = 'QBT_OBF_FLIP_RATE',
    boundary_rate_env: str = 'QBT_OBF_BOUNDARY_RATE',
    local_rate_env: str = 'QBT_OBF_LOCAL_RATE',
) -> Tuple[str, List[SU4]]:
    """
    Fold random single-qubit gates and output flips into a sequence of SU4s.
    """
    if len(unis) == 0:
        return target_state, unis

    if flip_rate_override is not None:
        flip_rate = float(flip_rate_override)
    else:
        try:
            flip_rate = float(os.getenv(flip_rate_env, '0.15'))
        except ValueError:
            flip_rate = 0.15
    try:
        boundary_rate = float(os.getenv(boundary_rate_env, '0.25'))
    except ValueError:
        boundary_rate = 0.25
    try:
        local_rate = float(os.getenv(local_rate_env, '0.25'))
    except ValueError:
        local_rate = 0.25
    swap_rate = 0.0

    rng = np.random.Generator(np.random.PCG64(seed ^ 0xA5A5A5A5))

    num_qubits = max(max(u.target0, u.target1) for u in unis) + 1
    occurrences = _build_occurrences(unis, num_qubits)

    touched = np.zeros(len(unis), dtype=bool)

    x_mask = (rng.random(num_qubits) < flip_rate).astype(np.uint8)
    X = _pauli_x()
    for q in np.nonzero(x_mask)[0]:
        idxs = occurrences[q]
        if len(idxs) == 0:
            continue
        idx = idxs[-1]
        su = unis[idx]
        left_leg = (q == su.target0)
        left_mul = _left_right_kron(X if left_leg else None, X if not left_leg else None)
        su.mat = (left_mul @ su.mat).astype(np.complex128, copy=False)
        touched[idx] = True

    if x_mask.any():
        target_state = _xor_bitstring(target_state, x_mask)

    processed_pairs: set[tuple[int, int]] = set()
    for q in range(num_qubits):
        idxs = occurrences[q]
        for a, b in zip(idxs, idxs[1:]):
            if rng.random() >= boundary_rate:
                continue
            key = (a, b) if a < b else (b, a)
            if key in processed_pairs:
                continue
            G = _random_u2(rng)
            Gdg = G.conj().T

            ua = unis[a]
            left_leg_a = (q == ua.target0)
            left_mul_a = _left_right_kron(Gdg if left_leg_a else None, Gdg if not left_leg_a else None)
            ua.mat = (left_mul_a @ ua.mat).astype(np.complex128, copy=False)

            ub = unis[b]
            left_leg_b = (q == ub.target0)
            right_mul_b = _left_right_kron(G if left_leg_b else None, G if not left_leg_b else None)
            ub.mat = (ub.mat @ right_mul_b).astype(np.complex128, copy=False)

            touched[a] = True
            touched[b] = True
            processed_pairs.add(key)


    for i, su in enumerate(unis):
        def _neighbor_pairs(idx_self: int) -> list[tuple[int, int, bool]]:
            cand: list[tuple[int, int, bool]] = []
            for q, leg_is_left in ((su.target0, True), (su.target1, False)):
                idxs = occurrences[q]
                if not idxs:
                    continue
                pos = None
                for t, idx in enumerate(idxs):
                    if idx == idx_self:
                        pos = t
                        break
                if pos is None:
                    continue
                if pos > 0:
                    cand.append((idxs[pos - 1], q, leg_is_left))
                if pos + 1 < len(idxs):
                    cand.append((idxs[pos + 1], q, leg_is_left))
            return cand

        if ensure_touch_all and not touched[i]:
            cand_pairs = _neighbor_pairs(i)
            cand_pairs_untouched = [(nb_idx, q, leg_is_left) for (nb_idx, q, leg_is_left) in cand_pairs if not touched[nb_idx]]
            choices = cand_pairs_untouched if cand_pairs_untouched else []

            if choices:
                nb_idx, q, leg_is_left = choices[rng.integers(0, len(choices))]
                G = _random_u2(rng)
                Gdg = G.conj().T

                left_mul_i = _left_right_kron(Gdg if leg_is_left else None, Gdg if not leg_is_left else None)
                su.mat = (left_mul_i @ su.mat).astype(np.complex128, copy=False)

                nb = unis[nb_idx]
                leg_is_left_nb = (q == nb.target0)
                right_mul_nb = _left_right_kron(G if leg_is_left_nb else None, G if not leg_is_left_nb else None)
                nb.mat = (nb.mat @ right_mul_nb).astype(np.complex128, copy=False)

                touched[i] = True
                touched[nb_idx] = True

        if not touched[i] and not ensure_touch_all and local_rate > 0.0 and rng.random() < local_rate:
            cand_pairs = _neighbor_pairs(i)
            if cand_pairs:
                nb_idx, q, leg_is_left = cand_pairs[rng.integers(0, len(cand_pairs))]
                G = _random_u2(rng)
                Gdg = G.conj().T
                left_mul_i = _left_right_kron(Gdg if leg_is_left else None, Gdg if not leg_is_left else None)
                su.mat = (left_mul_i @ su.mat).astype(np.complex128, copy=False)
                nb = unis[nb_idx]
                leg_is_left_nb = (q == nb.target0)
                right_mul_nb = _left_right_kron(G if leg_is_left_nb else None, G if not leg_is_left_nb else None)
                nb.mat = (nb.mat @ right_mul_nb).astype(np.complex128, copy=False)
                touched[i] = True
                touched[nb_idx] = True

    bt.logging.info(
        f"Obfuscation complete: touched {int(touched.sum())}/{len(unis)} SU4s, flips={int(x_mask.sum())}"
    )

    return target_state, unis


