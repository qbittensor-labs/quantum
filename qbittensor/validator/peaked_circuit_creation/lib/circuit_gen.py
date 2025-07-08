import warnings
from dataclasses import dataclass

import bittensor as bt
import cotengra as ctg
import numpy as np
import quimb.tensor as qtn
import torch
import tqdm
from torch import optim

from qbittensor.validator.peaked_circuit_creation.lib.circuit import (
    SU4, PeakedCircuit)
from qbittensor.validator.peaked_circuit_creation.peaked_circuits.functions import \
    range_unitary


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
    rqc_depth: int  # randomized circuit depth
    pqc_depth: int  # peaking circuit depth

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
        # if level < 0 or level > 5:
        #    raise ValueError("invalid difficulty level: must be 0 to 5")
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
        gen = np.random.Generator(np.random.PCG64(seed))
        target_state = "".join("1" if gen.random() < 0.5 else "0" for _ in range(self.nqubits))
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
        return PeakedCircuit.from_su4_series(target_state, peak_prob, unis, seed)


# Determine device for tensor operations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE} (CUDA available: {torch.cuda.is_available()})")

opti = ctg.ReusableHyperOptimizer(
    progbar=False,
    methods=["greedy"],
    reconf_opts={},
    max_repeats=36,
    optlib="optuna",
)


def norm_fn(psi: qtn.TensorNetwork):
    """
    Normalize the tensors in a network in place so that they are physically
    correct (i.e. unitary and obey the usual Born/probability rules).

    Args:
        psi (quimb.tensor.TensorNetwork):
            Tensor network whose tensors are to be normalized.
    """
    # parametrize our tensors as isometric/unitary
    return psi.isometrize(method="cayley")


def loss_fn(
    const: qtn.TensorNetwork,
    opt: qtn.TensorNetwork,
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
    return -abs((const.H & opt).contract(all, optimize=opti)) ** 2


class TNModel(torch.nn.Module):
    def __init__(self, opt, const):
        super().__init__()
        self.const = const
        params, self.skeleton = qtn.pack(opt)
        self.torch_params = torch.nn.ParameterDict(
            {
                # torch requires strings as keys
                str(i): torch.nn.Parameter(initial)
                for (i, initial) in params.items()
            }
        )

    def forward(self):
        # convert back to original int key format
        params = {int(i): p for (i, p) in self.torch_params.items()}
        # reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        return loss_fn(self.const, norm_fn(psi))


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
    range_unitary(
        psi, 0, 0, list(), depth, L - 1, "float64", seed_val, L - 1, uni_list=None, rand=True, start_layer=start_layer
    )
    return psi.astype_("complex128")


def make_torch(tn: qtn.TensorNetwork):
    """
    Convert all tensors in a network to complex torch tensors on `DEVICE`.
    """
    tn.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128, device=DEVICE))


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
    model = TNModel(opt, const)
    model()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message=".*trace might not generalize.*",
        )
        model = torch.jit.trace_module(model, {"forward": list()})
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    pbar = tqdm.tqdm(range(maxiters), disable=True)
    for step in pbar:
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"{loss:.6e}")
        # Log generation since progress bar turned off for pm2
        if step % 20 == 0:
            bt.logging.info(f"Circuit generation: Step {step}/{maxiters}, Current Loss: {loss:.6e}")

        # early stop if the peaking ratio is larger than the target
        if -loss * 2**nqubits > target_peaking:
            print(f"\nEarly stop: peak is >{target_peaking:g}x uniform")
            break

    # return the initial random circuit (without initial state tensors), the
    # peaking circuit, and the final probability of the target state
    opt = norm_fn(opt)
    rqc_tensors = list(init_rqc.H.tensors[nqubits:])
    pqc_tensors = list(opt.tensors[::-1])
    target_weight = float(-loss_fn(const, opt))
    return (rqc_tensors, pqc_tensors, target_weight)
