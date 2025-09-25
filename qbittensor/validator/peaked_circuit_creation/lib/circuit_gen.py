from dataclasses import dataclass
import time
import os

import bittensor as bt
import cotengra as ctg
import numpy as np
import quimb.tensor as qtn
import torch
from torch import optim

from qbittensor.validator.peaked_circuit_creation.lib.circuit import (
    SU4, PeakedCircuit)
import multiprocessing
from qbittensor.validator.peaked_circuit_creation.lib.obfuscate import (
    obfuscate_su4_series,
)
from qbittensor.validator.peaked_circuit_creation.lib.base_cache import (
    load_base_su4, save_base_su4,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
bt.logging.info(f"Using device: {DEVICE} (CUDA available: {torch.cuda.is_available()})")

opti = ctg.ReusableHyperOptimizer(
    progbar=False,
    methods=["greedy"],
    reconf_opts={},
    max_repeats=36,
    optlib="optuna",
)

def rand_gpu_unitary(n, dtype=torch.complex64, device=DEVICE):
    """Generates a random n x n unitary matrix on the GPU."""
    a = torch.randn(n, n, dtype=dtype, device=device)
    q, _ = torch.linalg.qr(a)
    return q

def range_unitary_gpu(
    psi, i_start, n_apply, list_u3, depth, n_Qbit, Qubit_ara, rand=True, start_layer=0
):
    """Applies random unitary gates to the tensor network on the GPU."""
    if n_Qbit <= 1:
        depth = 1

    for r in range(depth):
        is_even_layer = (r + start_layer) % 2 == 0
        start_idx = i_start
        end_idx = i_start + n_Qbit if is_even_layer else i_start + n_Qbit - 1
        
        for i in range(start_idx, end_idx, 2):
            q1 = i if is_even_layer else i + 1
            q2 = q1 + 1
            G = rand_gpu_unitary(4, device=DEVICE) if rand else torch.eye(4, dtype=torch.complex64, device=DEVICE)
            psi.gate_(G, (q1, q2), tags={"U", f"G{n_apply}", f"lay{Qubit_ara}", f"P{Qubit_ara}L{i}D{r}"})
            list_u3.append(f"G{n_apply}")
            n_apply += 1
    return n_apply, list_u3

def norm_fn(psi: qtn.TensorNetwork):
    """Isometrizes the tensors in a network in place."""
    return psi.isometrize(method="cayley")

def loss_fn(const: qtn.TensorNetwork, opt: qtn.TensorNetwork) -> torch.Tensor:
    """Computes the loss for optimization."""
    amplitude = (const.H & opt).contract(all, optimize=opti)
    return -torch.abs(amplitude) ** 2

class TNModel(torch.nn.Module):
    """A PyTorch module that wraps a quimb tensor network for optimization."""
    def __init__(self, opt_tn: qtn.TensorNetwork, const_tn: qtn.TensorNetwork):
        super().__init__()
        self.const = const_tn
        params, self.skeleton = qtn.pack(opt_tn)
        self.torch_params = torch.nn.ParameterDict({
            str(i): torch.nn.Parameter(initial_data)
            for i, initial_data in params.items()
        })

    def forward(self) -> torch.Tensor:
        params = {int(i): p for i, p in self.torch_params.items()}
        psi = qtn.unpack(params, self.skeleton)
        return loss_fn(self.const, norm_fn(psi))

def prepare_model_for_seed(target_state, rqc_depth, pqc_depth, seed):
    """
    Builds the initial tensor networks and TNModel for a given seed.
    This function is now deterministic based on the seed.
    """
    nqubits = len(target_state)
    
    torch.manual_seed(seed)
    
    init_rqc = make_qmps(nqubits * "0", rqc_depth, 0)
    target_pqc = make_qmps(target_state, pqc_depth, rqc_depth % 2)

    const = init_rqc & target_pqc.tensors[:nqubits]
    opt = qtn.TensorNetwork(target_pqc.tensors[nqubits:])

    model = TNModel(opt, const)
    model.to(DEVICE)
    
    return model, const, init_rqc, model.skeleton

def make_qmps(state: str, depth: int, start_layer: int) -> qtn.TensorNetwork:
    """Constructs a tensor network on the GPU. Relies on external torch.manual_seed."""
    L = len(state)
    psi = qtn.MPS_computational_state(state).astype_("complex64")
    psi.apply_to_arrays(lambda x: torch.from_numpy(x).to(device=DEVICE, dtype=torch.complex64))
    for k in range(L):
        psi[k].modify(left_inds=[f"k{k}"], tags=[f"I{k}", "MPS"])
    
    range_unitary_gpu(
        psi, i_start=0, n_apply=0, list_u3=[], depth=depth, n_Qbit=L - 1, 
        Qubit_ara=L - 1, rand=True, start_layer=start_layer
    )
    return psi
from contextlib import contextmanager


@contextmanager
def _temporary_env(name: str, value: str):
    prev = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if prev is not None:
            os.environ[name] = prev
        else:
            os.environ.pop(name, None)


def _build_unis_from_tensors(
    *,
    rqc_tensors: list[qtn.Tensor],
    pqc_tensors: list[qtn.Tensor],
    nqubits: int,
    pqc_depth: int,
    reverse_rqc: bool,
    reverse_pqc: bool,
) -> list[SU4]:
    rqc_seq = list(reversed(rqc_tensors)) if reverse_rqc else list(rqc_tensors)
    pqc_seq = list(reversed(pqc_tensors)) if reverse_pqc else list(pqc_tensors)
    unis: list[SU4] = []
    q0 = 0
    depth = 0
    for tens in rqc_seq:
        mat = tens.data.resolve_conj().cpu().numpy().reshape((4, 4))
        unis.append(SU4(q0, q0 + 1, mat))
        q0 += 2
        if q0 >= nqubits - 1:
            depth += 1
            q0 = depth % 2
    dshift = (nqubits + pqc_depth + 1) % 2
    q1 = nqubits - 1 - (depth + dshift) % 2
    for tens in pqc_seq:
        mat = tens.data.resolve_conj().cpu().numpy().reshape((4, 4))
        unis.append(SU4(q1 - 1, q1, mat))
        q1 -= 2
        if q1 <= 0:
            depth += 1
            q1 = nqubits - 1 - (depth + dshift) % 2
    return unis


def _generate_obfuscated_variants(
    *,
    target_state: str,
    base_unis: list[SU4],
    peak_prob: float,
    seed: int,
    total: int,
    ensure_touch_all: bool,
    pool=None,
) -> list[PeakedCircuit]:
    circuits: list[PeakedCircuit] = []
    for k in range(1, total):
        unis_k = [SU4(u.target0, u.target1, u.mat.copy()) for u in base_unis]
        salt = (seed ^ (k * 0x9E3779B1)) & 0xFFFFFFFF
        rng_obf = np.random.Generator(np.random.PCG64(int(salt) ^ 0xC0FFEE))
        flip_rate = 0.10 + 0.75 * float(rng_obf.random())
        with _temporary_env("QBT_OBF_SWAP_RATE", "0.0"):
            targ_k, unis_k = obfuscate_su4_series(
                target_state=target_state,
                unis=unis_k,
                seed=int(salt),
                flip_rate_override=flip_rate,
                ensure_touch_all=ensure_touch_all,
            )
        circuits.append(
            PeakedCircuit.from_su4_series(targ_k, peak_prob, unis_k, int(salt), pool=pool)
        )
    return circuits

def find_lucky_seed_and_make_circuit(
    target_state: str,
    rqc_depth: int,
    pqc_depth: int,
    base_seed: int,
    target_peaking: float = 1000.0,
) -> tuple[list[qtn.Tensor], list[qtn.Tensor], float]:
    """
    Pre-screens multiple seeds to find a "lucky" one, then runs the full
    optimization on that seed.
    """
    nqubits = len(target_state)
    
    NUM_SEEDS_TO_TRY = 20
    seed_losses = {}

    bt.logging.info(f"Pre-screening {NUM_SEEDS_TO_TRY} seeds by initial loss...")

    with torch.no_grad():
        for i in range(NUM_SEEDS_TO_TRY):
            candidate_seed = base_seed + i
            model, _, _, _ = prepare_model_for_seed(
                target_state, rqc_depth, pqc_depth, candidate_seed
            )
            
            initial_loss = model()
            
            seed_losses[candidate_seed] = initial_loss.item()

    best_seed = min(seed_losses, key=seed_losses.get)
    bt.logging.info(f" Selected champion seed: {best_seed} (Loss: {seed_losses[best_seed]:.4e})")

    # 1 forward-pass sanity check on the champion seed to catch bad
    # contraction cases (e.g., torch >25 dims) before full optim
    with torch.no_grad():
        try:
            _m_check, _, _, _ = prepare_model_for_seed(
                target_state, rqc_depth, pqc_depth, best_seed
            )
            _ = _m_check()
        except RuntimeError as e:
            bt.logging.warning(f"Champion seed {best_seed} failed prescreen forward: {e}")
            raise

    
    # Re-create the model from scratch with the best seed to start fresh
    model, const, init_rqc, skeleton = prepare_model_for_seed(
        target_state, rqc_depth, pqc_depth, best_seed
    )

    #lr = max(0.001, 10 ** (nqubits / 4 - 11))
    lr = 5e-2
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Main optimization loop
    maxiters = 1000
    start_opt_loop = time.perf_counter()
    
    for step in range(maxiters):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            bt.logging.debug(f"Step {step}/{maxiters}, Loss: {loss.item():.6e}")

        if -loss.item() * 2**nqubits > target_peaking:
            bt.logging.info(f"Early stop at step {step}: peak > {target_peaking:g}x uniform")
            break
            
    opt_loop_time = time.perf_counter() - start_opt_loop
    bt.logging.info(f"Optimization loop finished in {opt_loop_time:.4f} seconds.")

    with torch.no_grad():
        final_params = {int(i): p.data for i, p in model.torch_params.items()}
        opt_final = qtn.unpack(final_params, skeleton)
        opt_final_norm = norm_fn(opt_final)
        final_loss = loss_fn(const, opt_final_norm)
        target_weight = float(-final_loss.item())

    rqc_tensors = list(init_rqc.H.tensors[nqubits:])
    pqc_tensors = list(opt_final_norm.tensors[::-1])
    
    opti.cleanup()
    bt.logging.debug("Cleared cotengra optimizer cache")
    
    return rqc_tensors, pqc_tensors, target_weight


@dataclass
class CircuitParams:
    """High-level description of peaked circuit parameters."""
    difficulty: float
    nqubits: int
    rqc_depth: int
    pqc_depth: int

    @staticmethod
    def from_difficulty(level: float):
        safe = max(level + 3.9, 0.1)
        nqubits = int(round(12 + 10 * np.log2(safe)))
        nqubits = max(10, nqubits)
        rqc_mul = 150 * np.exp(-nqubits / 4) + 0.5
        rqc_depth = round(rqc_mul * nqubits)
        pqc_depth = max(1, nqubits // 5)
        return CircuitParams(level, nqubits, rqc_depth, pqc_depth)

    def compute_circuit(self, seed: int) -> PeakedCircuit:
        """Constructs a PeakedCircuit, now with the lucky seed search."""
        start_time = time.perf_counter()
        
        cached = load_base_su4(
            nqubits=self.nqubits,
            rqc_depth=self.rqc_depth,
            pqc_depth=self.pqc_depth,
            seed=seed,
        )
        if cached is not None:
            target_state, unis, peak_prob = cached
            return PeakedCircuit.from_su4_series(target_state, peak_prob, unis, seed)

        gen = np.random.Generator(np.random.PCG64(seed))
        target_state = "".join("1" if gen.random() < 0.5 else "0" for _ in range(self.nqubits))
        min_peak = float(os.getenv("QBT_MIN_PEAKING", "0"))
        peaking_threshold = max(min_peak, 10 ** (0.38 * self.difficulty + 2.102))

        # The `seed` is now the `base_seed` for the search
        (rqc, pqc, peak_prob) = find_lucky_seed_and_make_circuit(
            target_state,
            self.rqc_depth,
            self.pqc_depth,
            base_seed=seed, # The original seed starts the search
            target_peaking=peaking_threshold,
        )
        make_circuit_time = time.perf_counter() - start_time
        bt.logging.info(f"Total time for make_circuit: {make_circuit_time:.4f} seconds")

        start_conversion = time.perf_counter()
        r_rev = os.getenv("QBT_NORM_RQC_REVERSE", "0").strip() == "1"
        p_rev = os.getenv("QBT_NORM_PQC_REVERSE", "0").strip() == "1"

        unis = _build_unis_from_tensors(
            rqc_tensors=rqc,
            pqc_tensors=pqc,
            nqubits=self.nqubits,
            pqc_depth=self.pqc_depth,
            reverse_rqc=r_rev,
            reverse_pqc=p_rev,
        )
        conversion_time = time.perf_counter() - start_conversion
        bt.logging.info(f"Time for final conversion to SU4 (GPU->CPU): {conversion_time:.4f} seconds")

        try:
            save_base_su4(
                nqubits=self.nqubits,
                rqc_depth=self.rqc_depth,
                pqc_depth=self.pqc_depth,
                seed=seed,
                target_state=target_state,
                unis=unis,
                peak_prob=float(peak_prob),
            )
        except Exception:
            pass

        return PeakedCircuit.from_su4_series(target_state, peak_prob, unis, seed)

    def compute_circuits(self, seed: int, n_variants: int = 10) -> list[PeakedCircuit]:
        """Generate multiple circuit variants from one optimization run.

        Produces one original circuit plus (n_variants-1) obfuscated variants,
        all derived from the same optimized 4x4 unitaries.
        """
        start_time = time.perf_counter()

        cached = load_base_su4(
            nqubits=self.nqubits,
            rqc_depth=self.rqc_depth,
            pqc_depth=self.pqc_depth,
            seed=seed,
        )
        if cached is not None:
            target_state, unis, peak_prob = cached
            circuits: list[PeakedCircuit] = []
            circuits.append(PeakedCircuit.from_su4_series(target_state, peak_prob, unis, seed))
            total = max(1, int(n_variants))
            circuits.extend(
                _generate_obfuscated_variants(
                    target_state=target_state,
                    base_unis=unis,
                    peak_prob=peak_prob,
                    seed=seed,
                    total=total,
                    ensure_touch_all=True,
                )
            )
            return circuits

        gen = np.random.Generator(np.random.PCG64(seed))
        target_state = "".join("1" if gen.random() < 0.5 else "0" for _ in range(self.nqubits))
        min_peak = float(os.getenv("QBT_MIN_PEAKING", "0"))
        peaking_threshold = max(min_peak, 10 ** (0.38 * self.difficulty + 2.102))

        rqc, pqc, peak_prob = find_lucky_seed_and_make_circuit(
            target_state,
            self.rqc_depth,
            self.pqc_depth,
            base_seed=seed,
            target_peaking=peaking_threshold,
        )
        make_circuit_time = time.perf_counter() - start_time
        bt.logging.info(f"Total time for make_circuit (shared): {make_circuit_time:.4f} seconds")

        start_conversion = time.perf_counter()

        r_rev = os.getenv("QBT_NORM_RQC_REVERSE", "0").strip() == "1"
        p_rev = os.getenv("QBT_NORM_PQC_REVERSE", "0").strip() == "1"

        unis = _build_unis_from_tensors(
            rqc_tensors=rqc,
            pqc_tensors=pqc,
            nqubits=self.nqubits,
            pqc_depth=self.pqc_depth,
            reverse_rqc=r_rev,
            reverse_pqc=p_rev,
        )
        conversion_time = time.perf_counter() - start_conversion
        bt.logging.info(f"Time for final conversion to SU4 (GPU->CPU): {conversion_time:.4f} seconds")

        def _pool_initializer():
            import os as _os
            import signal as _signal
            _os.environ.setdefault('OMP_NUM_THREADS', '1')
            _os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
            _os.environ.setdefault('MKL_NUM_THREADS', '1')
            _signal.signal(_signal.SIGTERM, _signal.SIG_IGN)

        circuits: list[PeakedCircuit] = []
        pool = None
        try:
            pool = multiprocessing.Pool(initializer=_pool_initializer)
            circuits.append(PeakedCircuit.from_su4_series(target_state, peak_prob, unis, seed, pool=pool))

            total = max(1, int(n_variants))
            circuits.extend(
                _generate_obfuscated_variants(
                    target_state=target_state,
                    base_unis=unis,
                    peak_prob=peak_prob,
                    seed=seed,
                    total=total,
                    ensure_touch_all=True,
                    pool=pool,
                )
            )
        finally:
            if pool is not None:
                pool.close()
                pool.join()
                pool.terminate()

        try:
            save_base_su4(
                nqubits=self.nqubits,
                rqc_depth=self.rqc_depth,
                pqc_depth=self.pqc_depth,
                seed=seed,
                target_state=target_state,
                unis=unis,
                peak_prob=float(peak_prob),
            )
        except Exception:
            pass

        return circuits