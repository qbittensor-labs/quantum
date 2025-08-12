from copy import copy
from enum import IntEnum
import gc
import sys
from typing import List, Optional, Tuple

import cotengra as ctg
import numpy as np
import quimb as qu
import quimb.tensor as qtn
import torch
from torch import optim
import torch_optimizer
import tqdm
import warnings

from .circuit_meta import (Circuit, GateTensor, MPS)

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

def normalize(tn: qtn.TensorNetwork):
    """
    Normalize the tensors in a network in place so that they are physically
    correct (i.e. unitary and obey the usual Born/probability rules).

    Args:
        psi (quimb.tensor.TensorNetwork):
            Tensor network whose tensors are to be normalized.
    """
    # parameterize tensors as isometric/unitary
    return tn.isometrize(method="cayley")

def overlap(
    const: qtn.TensorNetwork,
    opt: qtn.TensorNetwork,
) -> qtn.Tensor | torch.Tensor:
    """
    Compute the overlap probability of a circuit + fixed input state with a
    particular target state.

    Args:
        const (quimb.tensor.TensorNetwork):
            The network of "constant" tensors containing:
                - the input state to the circuit
                - the randomizing circuit
                - the targeted basis state for peaking
        opt (quimb.tensor.TensorNetwork):
            The network of optimized tensors, containing:
                - the peaking circuit

    Returns:
        state_prob (quimb.tensor.Tensor or torch.Tensor):
            Scalar giving the overlap probability with the target peaked state
            as a (rank-0) tensor.
    """
    try:
        a = (const.H & opt).contract(all, optimize=opti)
        return abs(a) ** 2
    except Exception as e:
        print("= const indices =")
        for tens in const.tensors:
            print(tens.inds)
        print("= opt indices =")
        for tens in opt.tensors:
            print(tens.inds)
        print("free indices:", a.inds)
        raise e

def loss(
    const: qtn.TensorNetwork,
    opt: qtn.TensorNetwork,
) -> qtn.Tensor | torch.Tensor:
    """
    Thin wrapper around `overlap`, only applying a minus sign for optimization
    purposes.
    """
    return -overlap(const, opt)

class TNModel(torch.nn.Module):
    def __init__(self, const, opt):
        super().__init__()
        self.const = const
        params, self.skeleton = qtn.pack(opt)
        self.torch_params = torch.nn.ParameterDict({
            str(i): torch.nn.Parameter(initial)
            for (i, initial) in params.items()
        })

    def forward(self):
        params = { int(i): p for (i, p) in self.torch_params.items() }
        opt = qtn.unpack(params, self.skeleton)
        return loss(self.const, normalize(opt))

class OptimResult(IntEnum):
    """
    Termination condition for `optim_subcircuit`:
        - Success: terminated after crossing the target threshold.
        - LocalMinimum: terminated after getting stuck in a local minimum (i.e.
          the change in the objective function is less than some epsilon as a
          proportion of the current value of the objective function).
        - Maxiters: reached maxiters.
    """
    Success = 0
    LocalMinimum = 1
    Maxiters = 2

def _to_gpu(x: torch.Tensor) -> torch.Tensor:
    """
    Copy a tensor to the GPU.
    """
    return x.cuda()

def _rm_gpu(x: torch.Tensor) -> None:
    """
    Remove a tensor from the GPU.
    """
    with torch.no_grad():
        x.detach()
        del x
    return None

def _into_cpu(x: torch.Tensor) -> torch.Tensor:
    """
    Move a tensor to the CPU (being sure to `del` the GPU tensor).
    """
    with torch.no_grad():
        cpu = x.detach().cpu()
        del x
    return cpu

def _cuda_mem() -> Tuple[float, float]:
    allocated = torch.cuda.memory_allocated() / 1e3
    cached = torch.cuda.memory_reserved() / 1e3
    return (allocated, cached)

# def optim_circuit(
#     circuit: Circuit,
#     target_state: str,
#     pqc_prop: float,
#     target_peak: float,
#     maxiters: int = 2000,
#     epsilon: float = 1e-6,
# ):
#     """
#     Perform gradient descent optimization on the tensors of a peaking circuit
#     *in place*: The underlying tensor data of circuit elements will be modified,
#     leaving them in a state such that there will be peaking in the output of the
#     circuit when applied to the all-zero input state.
#
#     Args:
#         circuit (.circuit_meta.Circuit):
#             The circuit whose gates are to be optimized over.
#         target_state (str):
#             Output basis state on which to accumulate a peak. Shoud be a string
#             comprising only '0's and '1's.
#         pqc_prop (float):
#             Proportion (i.e. in the `[0, 1]` range) of each tile to optimize as
#             a "peaking" gate. This proportion is rounded up so that at least one
#             is always considered unless `pqc_prop == 0`.
#         target_peak (float > 0):
#             Target peaking probability.
#         maxiters (int > 0):
#             Maximum number of gradient descent iterations for each individual
#             tile optimization.
#         epsilon (float):
#             Terminate optimization for a tile if the magnitude of the change in
#             the overlap with the target state is less than this value, as a
#             proportion of the current overlap.
#
#     Returns:
#         peak_prob (float):
#             Final overlap probability of the output peaked state with
#             `target_state`.
#     """
#     input_state = MPS(len(target_state) * "0")
#     for gate in circuit.pre_gates:
#         input_state.apply_gate(gate)
#
#     # loop over tile ranks for individual optimization
#     #
#     # have to track the "local depth" of each qubit as gates are optimized and
#     # then applied to the state because we're working with non-rectangular
#     # subcircuits
#     local_depth: List[int] = len(target_state) * [0]
#     gate_app = 0
#     print(f"{circuit.shape.num_tile_ranks()} RANKS:")
#     for rank in range(circuit.shape.num_tile_ranks()):
#         print(f"RANK {rank} ({circuit.shape.tiles_in_rank(rank)} tiles)")
#         # if CUDA is available, move the input state to the GPU. We'll be
#         # moving it back to the CPU at the end
#         if DEVICE == "cuda":
#             input_state.mps.apply_to_arrays(_to_gpu)
#
#         tile_gates = circuit.tiles(rank)
#         # TODO: maybe possible to parallelize this loop
#         # TODO: check the numbers on the per-tile target threshold
#         target_threshold = target_peak ** (1 / len(tile_gates))
#         # target_threshold = target_peak
#         for tile in tile_gates:
#             if len(tile) < 1:
#                 continue
#             # pre-advance `local_depth` so that we have the right depths for the
#             # target state set of tensors
#             for gate in tile:
#                 layer = gate.layer()
#                 (ql, qr) = gate.qubits()
#                 local_depth[ql] = max(local_depth[ql], layer + 1)
#                 local_depth[qr] = max(local_depth[qr], layer + 1)
#             # remove 'k's from MPS physical indices so that they'll agree with
#             # those from the gates
#             input_state._remove_k()
#             target_mps = MPS.with_depths(target_state, local_depth)
#             target_mps._remove_k()
#
#             # move to GPU if available
#             if DEVICE == "cuda":
#                 for gate in tile:
#                     gate.tensor.apply_to_arrays(_to_gpu)
#                 target_mps.mps.apply_to_arrays(_to_gpu)
#
#             # partition tile into RQC and PQC
#             pqc_part = int(np.ceil(pqc_prop * len(tile)))
#             rqc = [gate.tensor for gate in tile[:-pqc_part]]
#             pqc = [gate.tensor for gate in tile[-pqc_part:]]
#
#             _optim_subcircuit(
#                 input_state,
#                 target_mps,
#                 rqc,
#                 pqc,
#                 target_threshold,
#                 maxiters,
#                 epsilon,
#             )
#
#             # conjugate pqc gates and isometrize together with rqc gates to
#             # properly preserve normalization
#             iso = normalize(
#                 qtn.TensorNetwork(rqc + [tens.H for tens in pqc]))
#             for (gate, iso_tens) in zip(tile, iso.tensors):
#                 gate.tensor.modify(data=iso_tens.data)
#
#             # update the input state with optimized gates
#             input_state._prepend_k()
#             for gate in tile:
#                 input_state.apply_gate(gate)
#                 gate_app += 1
#
#             # move tensors back the cpu to avoid huge GPU memory requirements
#             if DEVICE == "cuda":
#                 target_mps.mps.apply_to_arrays(_rm_gpu)
#                 del target_mps
#                 for gate in tile:
#                     gate.tensor.apply_to_arrays(_into_cpu)
#             print()
#
#         if DEVICE == "cuda":
#             input_state.mps.apply_to_arrays(_into_cpu)
#             # with torch.no_grad():
#             #     torch.cuda.empty_cache()
#             # gc.collect()
#         print()
#
#     # final evaluation of the target state overlap probability
#     target_mps = MPS.with_depths(target_state, local_depth)
#     if DEVICE == "cuda":
#         input_state.mps.apply_to_arrays(_to_gpu)
#         target_mps.mps.apply_to_arrays(_to_gpu)
#     self_overlap = float(overlap(input_state.mps, input_state.mps))
#     peak_prob = float(overlap(input_state.mps, target_mps.mps))
#     if DEVICE == "cuda":
#         input_state.mps.apply_to_arrays(_into_cpu)
#         target_mps.mps.apply_to_arrays(_rm_gpu)
#     print("self overlap prob:", self_overlap)
#     print("final overlap prob:", peak_prob)
#     return peak_prob

def optim_circuit_indep(
    circuit: Circuit,
    pqc_prop: float,
    target_peak: float,
    maxiters: int = 2000,
    epsilon: float = 1e-6,
) -> str:
    """
    Perform gradient descent optimization on the tensors of a peaking circuit
    *in place*: The underlying tensor data of circuit elements will be modified,
    leaving them in a state such that there will be peaking in the output of the
    circuit when applied to the all-zero input state.

    Args:
        circuit (.circuit_meta.Circuit):
            The circuit whose gates are to be optimized over.
        pqc_prop (float):
            Proportion (i.e. in the `[0, 1]` range) of each tile to optimize as
            a "peaking" gate. This proportion is rounded up so that at least one
            gate is always considered unless `pqc_prop == 0`.
        target_peak (float > 0):
            Target peaking probability.
        maxiters (int > 0):
            Maximum number of gradient descent iterations for each individual
            tile optimization.
        epsilon (float):
            Terminate optimization for a tile if the magnitude of the change in
            the overlap with the target state is less than this value, as a
            proportion of the current overlap.

    Returns:
        output_state (str):
            Peaked output state as a string of '0's and '1's.
        peak_prob (float):
            Estimate of the final overlap probability of the output peaked state
            with `target_state`.
    """
    gen = np.random.Generator(np.random.PCG64(circuit.seed))
    # throughout, approximate the output of each peaking circuit as a product
    # state, and maintain an estimate of the peaking by simply multipling the
    # peaking probabilities
    #
    # use a List[bool] for slicing and mutability
    pstate = circuit.shape.nqubits * [False]
    peak_prob = 1.0

    # loop over tile ranks for individual optimization
    # have to track the "local depth of each qubit as gates are optimized in
    # order to property construct MPSs because we're working with
    # non-rectangular subcircuits
    local_depth = circuit.shape.nqubits * [0]
    num_tiles = circuit.shape.num_tiles()
    local_peaking_target = target_peak ** (1 / num_tiles)
    print(f"LOCAL PEAKING TARGET = {local_peaking_target:g}")
    print(f"{circuit.shape.num_tile_ranks()} RANKS:")
    tile_count = 0
    for rank in range(circuit.shape.num_tile_ranks()):
        print(f"RANK {rank} ({circuit.shape.tiles_in_rank(rank)} tiles)")
        tile_gates = circuit.tiles(rank)

        # TODO: maybe possible to parallelize this loop
        for tile in tile_gates:
            if len(tile) < 1:
                continue
            (q0, q1) = _qubit_bounds(tile)
            qubit_len = q1 - q0

            # make the local input state MPS
            input_bits = "".join("1" if b else "0" for b in pstate[q0:q1])
            input_state = MPS.with_depths(
                input_bits, local_depth[q0:q1], ind_offs=q0, with_k=False)

            # pre-advance `local_depth` so that we have the right depths for the
            # target state set of tensors
            for gate in tile:
                layer = gate.layer()
                (ql, qr) = gate.qubits()
                local_depth[ql] = max(local_depth[ql], layer + 1)
                local_depth[qr] = max(local_depth[qr], layer + 1)

            # make the target state MPS
            target_bits = "".join(
                "0" if gen.random() < 0.5 else "1" for _ in range(qubit_len))
            target_state = MPS.with_depths(
                target_bits, local_depth[q0:q1], ind_offs=q0, with_k=False)

            # move to GPU if available
            if DEVICE == "cuda":
                input_state.mps.apply_to_arrays(_to_gpu)
                target_state.mps.apply_to_arrays(_to_gpu)
                for gate in tile:
                    gate.tensor.apply_to_arrays(_to_gpu)

            # partition tile into RQC and PQC
            # TODO: maybe find a better way to partition here -- if we're
            # targeting only low peaking, a minimal set of tensors to optimize
            # over would be any set with full connectivity over the qubits in
            # the tile
            pqc_part = int(np.ceil(pqc_prop * len(tile)))
            rqc = [gate.tensor for gate in tile[:-pqc_part]]
            pqc = [gate.tensor for gate in tile[-pqc_part:]]

            # doit
            # use an adjustment factor
            #   local_peaking_target ** tile_count / peak_prob
            # to "catch up" for possible smaller peaking in previous tiles
            target_threshold = local_peaking_target ** (tile_count + 1) / peak_prob
            (_, subpeak_loss) = _optim_subcircuit(
                input_state,
                target_state,
                rqc,
                pqc,
                target_threshold,
                maxiters,
                epsilon,
            )
            tile_count += 1

            # conjugate pqc gates and isometrize together with rqc gates to
            # property preserve normalization
            iso = normalize(
                qtn.TensorNetwork(rqc + [tens.H for tens in pqc]))
            for (gate, iso_tens) in zip(tile, iso.tensors):
                gate.tensor.modify(data=iso_tens.data)

            # update `pstate` and the total peaking estimate
            for (q, b) in zip(range(q0, q1), target_bits):
                pstate[q] = bool(int(b))
            peak_prob *= -subpeak_loss

            # move tensors back to the CPU and clear GPU cache to avoid huge GPU
            # memory requirements
            if DEVICE == "cuda":
                input_state.mps.apply_to_arrays(_rm_gpu)
                del input_state
                target_state.mps.apply_to_arrays(_rm_gpu)
                del target_state
                for gate in tile:
                    gate.tensor.apply_to_arrays(_into_cpu)
            print()

    output_state = "".join("1" if b else "0" for b in pstate)
    return (output_state, peak_prob)

def _qubit_bounds(tile: List[GateTensor]) -> Optional[slice]:
    """
    Return the `(min, max + 1)` slicing bounds for the contiguous range of
    qubits on which a gate tile acts.
    """
    qmin = None
    qmax = None
    for gate in tile:
        (l, r) = gate.qubits()
        qmin = l if qmin is None else min(qmin, l)
        qmax = r if qmax is None else max(qmax, r)
    return None if qmin is None or qmax is None else (qmin, qmax + 1)

def _optim_subcircuit(
    input_state: qtn.MatrixProductState,
    target_state: qtn.MatrixProductState,
    rqc: List[qtn.Tensor],
    pqc: List[qtn.Tensor],
    target_threshold: float,
    maxiters: int = 1000,
    epsilon: float = 1e-3,
) -> Tuple[OptimResult, float]:
    """
    Perform gradient descent optimization on the tensors of a peaking subcircuit
    *in place*: The underlying tensor data of the elements of `pqc` will be
    modified, leaving them in a state such that there will be peaking within the
    subset of the input MPS they act on. All relevant tensors should already be
    stored in the GPU, if available.

    Args:
        input_state (quimb.tensor.TensorNetwork):
            Handle to the MPS input to the subcircuit, will not be modified.
            This should be the entire N-qubit state.
        target_state (.circuit_meta.MPS):
            Handle to the target MPS basis state, will not be modified. This
            should be the entire N-qubit basis state.
        rqc (List[.circuit_meta.GateTensor]):
            List of tensors forming the randomizing part of the subcircuit, will
            not be modified. These are allowed to act on only a subset of
            qubits.
        pqc (List[.circuit_meta.GateTensor]):
            List of tensors forming the peaking part of the subcircuit; the data
            in the underlying tensors (i.e. `GateTensor.tensor.data`) will be
            modified such that some amount of peaking will occur.
        maxiters (int):
            Limit optimization to this maximum number of steps.
        target_threshold (float):
            Terminate optimization early if the overlap with the target state is
            at least this value. Values greater than 1 guarantee that
            optimization will be performed over the full `maxiters` steps and
            non-positive values disable optimization.
        epsilon (float):
            Terminate optimization if the magnitude of the change in the overlap
            with the target state is less than this value, as a proportion of
            the current overlap.

    Returns:
        term (OptimResult):
            Termination condition.
        loss (float):
            Final loss function value.
    """
    nqubits = len(input_state.mps.tensors)
    # partition tensors:
    #   * `const` is the input and target states with randomizing circuit
    #   * `opt` is the peaking circuit, which needs to be optimized
    const = qtn.TensorNetwork(
        list(input_state.mps.tensors)
        + list(target_state.mps.tensors)
        + rqc
    )
    opt = qtn.TensorNetwork(pqc)

    # now do tensor network optimization to actually peak the target state
    model = TNModel(const, opt)
    model()

    # JIT tracing disabled to prevent memory issues
    # scale the learning rate based on qubit count to prevent low peaking for
    # higher diff
    # TODO: make this scaling based on difficulty, not qubits
    optimizer = optim.AdamW(
        model.parameters(),
        lr=max(0.001, 10 ** (nqubits / 4 - 11)),
        # lr=0.001,
    )
    pbar = tqdm.tqdm(range(maxiters), disable=False)
    prev_loss: Optional[torch.Tensor] = None
    res: Optional[OptimResult] = None
    for step in pbar:
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"{loss:.6e}")
        # if step % 100 == 0:
        #     # bt.logging.info(
        #     print(
        #         f"Circuit generation: step {step}/{maxiters}; "
        #         + f"current loss: {loss:.6e}"
        #     )
        # early stop if loss threshold is crossed or if the change in loss is
        # too small
        if -loss > target_threshold:
            res = OptimResult.Success
            break
        if prev_loss is not None and abs((loss - prev_loss) / loss) < epsilon:
            res = OptimResult.LocalMinimum
            break
        prev_loss = loss
    if res is None:
        print(f"Reached maxiters ({maxiters}); final loss: {loss:.6e}")
        res = OptimResult.Maxiters
    elif res == OptimResult.Success:
        print(f"Early stop: peak is >{target_threshold:g} (step {step}/{maxiters})")
    elif res == OptimResult.LocalMinimum:
        print(f"Early stop: change in loss is too small; loss: {loss:.6e}")

    # clean up: clear cotengra optimizer cache
    opti.cleanup()
    model.cpu()
    del model
    del optimizer
    # bt.logging.debug("Cleared cotengra optimizer cache")

    return (res, float(loss))

