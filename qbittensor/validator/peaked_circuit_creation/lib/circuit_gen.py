from dataclasses import dataclass
from itertools import product
from multiprocessing import Pool
from typing import Optional
import cotengra as ctg
import numpy as np
import quimb as qu
import quimb.tensor as qtn
import torch
from torch import optim
import warnings
import torch_optimizer
import bittensor as bt
from qbittensor.validator.peaked_circuit_creation.peaked_circuits.functions import range_unitary
from qbittensor.validator.peaked_circuit_creation.lib.circuit import (PeakedCircuit, SU4)

PROJ_EVERY = 8
NORM_METHOD_FAST = "qr"
NORM_METHOD_STRICT = "qr"

# Kind of janky workaround some quimb/autoray paths call backend 'sign' on complex tensors.
# torch mapped 'sign' to 'sgn' for complex to avoid NotImplemented
try:
    import autoray as _ar
    _ar.register_function('torch', 'sign', torch.sgn)
except Exception:
    _orig_sign = torch.sign
    def _patched_sign(x):
        if torch.is_complex(x):
            return torch.sgn(x)
        return _orig_sign(x)
    torch.sign = _patched_sign 

@dataclass
class CircuitParams:
    """
    High-level description of peaked circuit parameters.

    Fields:
        difficulty (float):
            The difficulty level of the circuit, should be `0 <= level <= 5`.
        nqubits (int):
            The number of qubits, should be positive.
        rqc_depth (int):
            The number of randomizing circuit layers, should be positive.
        pqc_depth (int):
            The number of peaking circuit layers, should be positive.
    """
    difficulty: float
    nqubits: int
    rqc_depth: int # randomized circuit depth
    pqc_depth: int # peaking circuit depth

    @staticmethod
    def from_difficulty(level: float):
        """
        Determine the parameters for a circuit based on a single difficulty
        level, defined for `0 <= level <= 5`.

        Args:
            level (float):
                Floating-point difficulty level, with 0 being the lowest
                difficulty. Must be 0 to 5 (inclusive).

        Returns:
            params (CircuitParams):
                Data struct containing a number of qubits, RQC depth, and PQC
                depth for the difficulty level.

        Raises:
            ValueError if `level` is less than 0 or greater than 5.
        """
        if level < 0 or level > 5:
            raise ValueError("invalid difficulty level: must be 0 to 5")
        nqubits = int(12 + 10 * np.log2(level + 3.9))
        rqc_mul = 150 * np.exp(-nqubits / 4) + 0.5
        rqc_depth = round(rqc_mul * nqubits)
        pqc_depth = max(1, nqubits // 5)
        return CircuitParams(level, nqubits, rqc_depth, pqc_depth)

    def compute_circuit(self, seed: int) -> PeakedCircuit:
        """
        Construct a randomized `PeakedCircuit` according to `self`, with fixed
        seed. The value of `seed` determines only the target peaked state (i.e.
        a fixed seed will generate circuits that output states with peaking in
        the same target state, but the gates may be different).

        Args:
            seed (int):
                Seed value for circuit generation.

        Returns:
            circuit (PeakedCircuit):
                Output peaked circuit.
        """
        try:
            bt.logging.info(
                f"[peaked] starting circuit generation: nqubits={self.nqubits}, rqc_depth={self.rqc_depth}, pqc_depth={self.pqc_depth}"
            )
        except Exception:
            pass
        gen = np.random.Generator(np.random.PCG64(seed))
        target_state = "".join(
            "1" if gen.random() < 0.5 else "0" for _ in range(self.nqubits))
        peaking_threshold = max(20, 10 ** (0.38 * self.difficulty + 2.102))
        (rqc, pqc, peak_prob) = make_circuit(
            target_state,
            self.rqc_depth,
            self.pqc_depth,
            seed,
            target_peaking=peaking_threshold,
        )
        # convert tensors to ordinary 2D numpy arrays -- have to get qubit
        # indices right for brickwork circuits
        unis = list()
        q0 = 0
        depth = 0
        for tens in rqc:
            mat = tens.data.cpu().resolve_conj().numpy().reshape((4, 4))
            unis.append(SU4(q0, q0 + 1, mat))
            q0 += 2
            if q0 >= self.nqubits - 1:
                depth += 1
                q0 = depth % 2
        # the pqc tensors were generated in "backwards" (i.e. reversed time and
        # space) order; the time is properly reversed at the end of
        # `make_circuit`, but we need to deal with space here by reversing the
        # counting order for qubit indices
        dshift = (self.nqubits + self.pqc_depth + 1) % 2
        q1 = self.nqubits - 1 - (depth + dshift) % 2
        for tens in pqc:
            mat = tens.data.cpu().resolve_conj().numpy().reshape((4, 4))
            unis.append(SU4(q1 - 1, q1, mat))
            q1 -= 2
            if q1 <= 0:
                depth += 1
                q1 = self.nqubits - 1 - (depth + dshift) % 2
        return PeakedCircuit.from_su4_series(
            target_state, peak_prob, unis, seed)

# Determine device for tensor operations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
bt.logging.info(f"Using device: {DEVICE} (CUDA available: {torch.cuda.is_available()})")

_ctg_methods = ["greedy"]
try:
    import kahypar as _  # noqa: F401
    _ctg_methods = ["greedy", "kahypar"]
except Exception:
    _ctg_methods = ["greedy"]

opti = ctg.ReusableHyperOptimizer(
    progbar=False,
    methods=_ctg_methods,
    reconf_opts={},
    max_repeats=8,
    optlib="optuna",
    parallel=True,
)

def norm_fn(psi: qtn.TensorNetwork, method: str = NORM_METHOD_STRICT):
    """Normalize the tensors in-place with the given method and return psi."""
    psi.isometrize_(method=method)
    return psi

def loss_fn(
    const: qtn.TensorNetwork,
    opt: qtn.TensorNetwork,
    *,
    optimize_plan=None,
) -> qtn.Tensor | torch.Tensor:
    """
    Compute the loss function for peaked circuit optimization.

    Args:
        const (quimb.tensor.TensorNetwork):
            The network of "constant" tensors, containing:
                - the inital (all-zero) state
                - the randomizing circuit
                - the target output state
        opt (quimb.tensor.TensorNetwork):
            The network of optimized tensors, containing:
                - the peaking circuit

    Returns:
        state_prob (quimb.tensor.Tensor or torch.Tensor):
            Scalar giving the probability of the target output state, as a
            (rank-0) tensor.
    """
    contraction_opt = optimize_plan if optimize_plan is not None else opti
    return -abs((const.H & opt).contract(all, optimize=contraction_opt)) ** 2

class TNModel(torch.nn.Module):
    def __init__(self, opt, const, optimize_plan=None, profiler=None, proj_every: int = PROJ_EVERY):
        super().__init__()
        self.const = const
        params, self.skeleton = qtn.pack(opt)
        self.optimize_plan = optimize_plan
        self._profiler = profiler
        self._step = 0
        self.proj_every = max(1, int(proj_every))
        self.torch_params = torch.nn.ParameterDict({
            # torch requires strings as keys
            str(i): torch.nn.Parameter(initial)
            for (i, initial) in params.items()
        })

    def forward(self):
        # convert back to original int key format
        params = { int(i): p for (i, p) in self.torch_params.items() }
        # reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        if (self._step % self.proj_every) == 0:
            psi_n = norm_fn(psi, NORM_METHOD_STRICT)
        else:
            psi_n = norm_fn(psi, NORM_METHOD_FAST)
        out = loss_fn(self.const, psi_n, optimize_plan=self.optimize_plan)
        self._step += 1
        return out

 

def make_qmps(
    state: str,
    depth: int,
    start_layer: int,
    seed_val: int,
) -> qtn.TensorNetwork:
    """
    Construct a new tensor network corresponding to a matrix product state
    attached to a randomized brickwork circuit of depth `depth`. `start_layer`
    is used to offset the initial left-most gate in the randomized circuit.

    Args:
        state (str):
            Initial MPS state before the brickwork circuit. Every character in
            this string should be either '0' or '1'.
        depth (int):
            Depth of the randomized brickwork circuit.
        start_layer (int):
            Used to determine whether the left-most gate on the first layer
            starts at qubit 0 or qubit 1.
        seed_val (int):
            Initial seed value used to generate the brickwork gates.

    Returns:
        qmps (quimb.tensor.TensorNetwork):
            The resulting matrix product state with randomized brickwork
            circuit, as a tensor network.
    """
    L = len(state)
    psi = qtn.MPS_computational_state(state)
    for k in range(L):
        psi[k].modify(left_inds=[f"k{k}"], tags=[f"I{k}", "MPS"])
    qubit_ara = L - 1
    range_unitary(
        psi, 0, 0, list(), depth, L - 1, "float64", seed_val, L - 1,
        uni_list=None, rand=True, start_layer=start_layer)
    return psi.astype_('complex128')

def make_torch(tn: qtn.TensorNetwork):
    """
    Convert all tensors in a network to complex torch tensors on `DEVICE`.
    """
    tn.apply_to_arrays(
        lambda x: torch.tensor(x, dtype=torch.complex128, device=DEVICE))

def make_circuit(
    target_state: str,
    rqc_depth: int,
    pqc_depth: int,
    seed: int,
    target_peaking: float = 1000.0,
) -> tuple[list[qtn.Tensor], list[qtn.Tensor], float]:
    """
    Construct a brickwork peaking circuit producing `target_state` as its peaked
    output.

    Args:
        target_state (str):
            The computational basis state on which to produce a peak. Should be
            a string of only '0' or '1'.
        rqc_depth (int):
            The number of initial brickwork randomizing layers.
        pqc_depth (int):
            The number of peaking brickwork layers.
        seed (int):
            Used for initial sampling of the randomizing and peaking gates.
        target_peaking (float, optional):
            Terminate optimization when the ratio of the target state's
            probability to the N-qubit uniform probability (1/2^N) crosses this
            threshold.

    Returns:
        rqc (list[torch.Tensor]):
            Tensors corresponding to gates in the initial randomizing circuit.
            This list is ordered first by left-to-right qubit order, starting at
            the leftmost qubit (qubit 0), and then by increasing circuit depth.
            These tensors are returned conjugated; call `.resolv_conj()` to get
            the exact matrix elements of the gate.
        pqc (list[torch.Tensor]):
            Tensors corresponding to gates in the peaking circuit. This list is
            ordered first by *right-to-left* qubit order, with the starting
            rightmost qubit determined by the RQC depth, and then by increasing
            circuit depth.
        target_prob (float):
            The output probability of the target basis state.
    """
    # generate inital set of tensors:
    #   * `init_rqc[:nqubits]`: initial (all-zero state)
    #   * `init_rqc[nqubits:]`: randomizing circuit
    #   * `target_pqc[:nqubits]`: target state
    #   * `target_pqc[nqubits:]`: peaking circuit (to be optimized)
    nqubits = len(target_state)
    init_rqc = make_qmps(nqubits * "0", rqc_depth, 0, seed)
    make_torch(init_rqc)
    target_pqc = make_qmps(target_state, pqc_depth, rqc_depth % 2, seed)
    make_torch(target_pqc)

    # separate all the tensors corresponding to the input state, the target
    # state, and the randomized gates out into a network of "constants"
    const = init_rqc & target_pqc.tensors[:nqubits]
    # the rest are exactly the peaking gates, to be optimized below
    opt = qtn.TensorNetwork(target_pqc.tensors[nqubits:])

    # now do tensor network optimization to actually peak the target state
    maxiters = 1000
    # Here we precompute a contraction tree on a normalized copy of the TN structure,
    # then reuse it every iteration to avoid re-planning without changing math.
    opt_copy = opt.copy()
    norm_fn(opt_copy, NORM_METHOD_STRICT)
    try:
        contraction_tree = (const.H & opt_copy).contract(all, optimize=opti, get='tree')
    except Exception:
        bt.logging.info("contraction fallback: using 'greedy' plan once")
        contraction_tree = (const.H & opt_copy).contract(all, optimize='greedy', get='tree')
    model = TNModel(opt, const, optimize_plan=contraction_tree, proj_every=PROJ_EVERY)
    lr = 5e-2
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=300, gamma=0.5)
    previous_loss = torch.inf
    for step in range(maxiters):
        optimizer.zero_grad(set_to_none=True)
        loss = model()
        loss.backward()
        optimizer.step()
        # early stop if the peaking ratio is larger than the target
        if -loss * 2 ** nqubits > target_peaking:
            bt.logging.info(f"Early stop: peak is >{target_peaking:g}x uniform")
            break

    # return the initial random circuit (without initial state tensors), the
    # peaking circuit, and the final probability of the target state
    opt = norm_fn(opt, NORM_METHOD_STRICT)
    rqc_tensors = list(init_rqc.H.tensors[nqubits:])
    pqc_tensors = list(opt.tensors[::-1])
    target_weight = float(-loss_fn(const, opt))
    return (rqc_tensors, pqc_tensors, target_weight)

