"""
High-level tools to manage locations within brickwork circuits.

Here we want to consider partitioning the gates in the circuit into discrete,
non-overlapping "tiles" in order to allow peaked circuit generation to be
parallelized as smaller chunks (i.e. per tile). For this to work, each tile
should be a simple shape that tiles the plane and is convex in order to allow
horizontally adjacent tiles to be processed in parallel. The tiling we consider
here is based on a "rhombic" partitioning. For example, considering the
following circuit,

     |   |   |   |   |   |   |   |   |   |   |   |
     |  .-----. .-----. .-----. .-----. .-----.  |
     |  | 39  | | 40  | | 41  | | 42  | | 43  |  |
     |  '-----' '-----' '-----' '-----' '-----'  |
     |   |   |   |   |   |   |   |   |   |   |   |
    .-----. .-----. .-----. .-----. .-----. .-----.
    | 33  | | 34  | | 35  | | 36  | | 37  | | 38  |
    '-----' '-----' '-----' '-----' '-----' '-----'
     |   |   |   |   |   |   |   |   |   |   |   |
     |  .-----. .-----. .-----. .-----. .-----.  |
     |  | 28  | | 29  | | 30  | | 31  | | 32  |  |
     |  '-----' '-----' '-----' '-----' '-----'  |
     |   |   |   |   |   |   |   |   |   |   |   |
    .-----. .-----. .-----. .-----. .-----. .-----.
    | 22  | | 23  | | 24  | | 25  | | 26  | | 27  |
    '-----' '-----' '-----' '-----' '-----' '-----'
     |   |   |   |   |   |   |   |   |   |   |   |
     |  .-----. .-----. .-----. .-----. .-----.  |
     |  | 17  | | 18  | | 19  | | 20  | | 21  |  |
     |  '-----' '-----' '-----' '-----' '-----'  |
     |   |   |   |   |   |   |   |   |   |   |   |
    .-----. .-----. .-----. .-----. .-----. .-----.
    | 11  | | 12  | | 13  | | 14  | | 15  | | 16  |
    '-----' '-----' '-----' '-----' '-----' '-----'
     |   |   |   |   |   |   |   |   |   |   |   |
     |  .-----. .-----. .-----. .-----. .-----.  |
     |  |  6  | |  7  | |  8  | |  9  | | 10  |  |
     |  '-----' '-----' '-----' '-----' '-----'  |
     |   |   |   |   |   |   |   |   |   |   |   |
    .-----. .-----. .-----. .-----. .-----. .-----.
    |  0  | |  1  | |  2  | |  3  | |  4  | |  5  |
    '-----' '-----' '-----' '-----' '-----' '-----'
     |   |   |   |   |   |   |   |   |   |   |   |

with gates labeled by indices, we partition gates in the shape of (clipped)
rhombi of, e.g., three gates across:
    - { 0, 1, 2, 6, 7, 12 }
    - { 3, 4, 5, 9, 10, 15 }
    - { 11, 17, 22 }
    - { 8, 13, 14, 18, 19, 20, 24, 25, 30 }
    - { 16, 21, 27 }
    - ...
Under this partitioning, we say that each tile has a "rank" corresponding to its
rough depth in the circuit, with tiles tangent only at their left or right
corners having the same rank. In this example, the first two tiles lie at rank
0, while the next three are at rank 1, and so on. In this scheme, all tiles with
the same rank can be processed in parallel, and each rank can be mapped to an
ordinary gate layer in the circuit as `[layer] = [rank] * [tile width]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable, List, Self, Tuple

import numpy as np
import quimb as qu
import quimb.tensor as qtn
import torch

class Neighbor(IntEnum):
    UL = 0 # upper-left
    UR = 1 # upper-right
    LL = 2 # lower-left
    LR = 3 # lower-right

@dataclass
class CircuitShape:
    """
    Data class to manage indices for partitioned brickwork circuits.
    """
    nqubits: int
    depth: int
    tile_width: int

    def gates_in_layer(self, layer: int) -> int:
        """
        Return the number of gates in a given layer. `layer` is allowed to
        exceed the bounds of the circuit.
        """
        return (self.nqubits - layer % 2) // 2

    def layer_leader(self, layer: int) -> int:
        """
        Return the index of the left-most gate in a given layer. `layer` should
        be at least 0.
        """
        return sum(self.gates_in_layer(d) for d in range(layer))

    def gate_range(self, layer: int) -> range:
        """
        Return a range giving the indices of all gates in a given layer. `layer`
        should be at least 0.
        """
        lo = self.layer_leader(layer)
        hi = lo + self.gates_in_layer(layer)
        return range(lo, hi)

    def neighbor(
        self,
        ndir: Neighbor,
        layer: int,
        gate: int,
    ) -> Tuple[int, int]:
        """
        Return the (depth, index) position of the `ndir` neighbor of a gate in a
        specified layer.
        """
        corr = (self.nqubits % 2) * (layer % 2)
        corr2 = int(self.nqubits % 2 == 0)
        ngates0 = self.nqubits // 2
        if ndir == Neighbor.UL:
            return (layer + 1, gate + ngates0 + corr - 1)
        elif ndir == Neighbor.UR:
            return (layer + 1, gate + ngates0 + corr)
        elif ndir == Neighbor.LL:
            return (layer - 1, gate - ngates0 + corr + corr2 - 1)
        elif ndir == Neighbor.LR:
            return (layer - 1, gate - ngates0 + corr + corr2)
        else:
            raise ValueError(f"invalid neighbor direction `{ndir}`")

    def _neighbor_series(
        self,
        ndir: Neighbor,
        length: int,
        layer: int,
        gate: int,
    ) -> Iterable[Tuple[int, int]]:
        for _ in range(length):
            yield (layer, gate)
            (layer, gate) = self.neighbor(ndir, layer, gate)

    def num_tile_ranks(self) -> int:
        """
        Return the total number of tile ranks in the circuit.
        """
        if self.depth <= 0:
            return 0
        else:
            d = self.depth
            w = self.tile_width
            return int(np.ceil((d + w - 1) / w))

    def tile_layer(self, rank: int) -> int:
        """
        Return the ordinary gate layer corresponding to a given tile rank. This
        is the layer index of the single gate that forms the tile's left-most
        corner.
        """
        return rank * self.tile_width

    def tiles_in_rank(self, rank: int) -> int:
        """
        Return the number of tiles at a given rank, including tiles that are
        clipped by any circuit boundaries. `rank` is allowed to exceed the
        bounds of the circuit.
        """
        depth = self.tile_layer(rank)
        num_gates = self.gates_in_layer(depth)
        return int(np.ceil(num_gates / self.tile_width)) + rank % 2

    def _tile_leader(self, rank: int) -> int:
        layer_leader = self.layer_leader(self.tile_layer(rank))
        if rank % 2 == 1:
            return layer_leader - (self.tile_width // 2 + self.tile_width % 2)
        else:
            return layer_leader

    def tiles(self, rank: int) -> List[List[int]]:
        """
        Return a list of all gate indices in tiles at a given rank, grouped by
        tile. Tiles will be clipped by the boundaries of the circuit.
        """
        layer_range = range(self.depth)
        target_layer = self.tile_layer(rank)
        layer_ok = lambda d: d in layer_range

        # order the ranges so that we can index using negative integers,
        # corresponding to layer indices relative to `target_layer`
        gate_ranges = (2 * self.tile_width - 1) * [None]
        for d in range(-self.tile_width + 1, self.tile_width):
            gate_ranges[d] = self.gate_range(target_layer + d)
        gate_ok = lambda d, g: g in gate_ranges[d - target_layer]

        leader = self._tile_leader(rank)
        num_tiles = self.tiles_in_rank(rank)
        tiles = list()
        for k in range(num_tiles):
            iter_tile_ll = self._neighbor_series(
                Neighbor.LR,
                self.tile_width,
                target_layer,
                leader + k * self.tile_width,
            )
            tiles.append(sorted(
                gate
                for (layer0, gate0) in iter_tile_ll
                for (layer, gate) in self._neighbor_series(
                    Neighbor.UR, self.tile_width, layer0, gate0)
                if layer_ok(layer) and gate_ok(layer, gate)
            ))
        return tiles

    def num_tiles(self) -> int:
        """
        Return the total number of tiles in the circuit.
        """
        return sum(self.tiles_in_rank(r) for r in range(self.num_tile_ranks()))

    def sample_gates(self, seed: int) -> Circuit:
        """
        Fill `self` with randomly sampled gates.
        """
        return Circuit.from_shape(self, seed)


def rand_unis_2q(seed: int) -> Iterable[qu.qarray]:
    """
    Return an infinite generator over Haar-random two-qubit operators. Seeding
    is performed globally via `quimb.seed_rand`, so beware of side effects via
    multiple calls to this function or other calls to `quimb.seed_rand` in
    between yielded elements.
    """
    qu.seed_rand(seed)
    while True:
        yield qu.rand_uni(4, dtype=complex)

class GateTensor:
    """
    Newtype over a `quimb.tensor.Tensor` to ensure a particular index labeling
    scheme. Each underlying `quimb.tensor.Tensor` is backed by a `torch.Tensor`
    (rather than a `quimb.qarray`).
    """
    tensor: qtn.Tensor

    def __init__(self, nqubits: int, layer: int, left: int, mat: qu.qarray):
        """
        Construct a new `GateTensor` living in the `layer`-th layer of a circuit
        and being applied to the `left`-th and `left + 1`-th qubits.
        """
        layer_l = max(0, layer - 1) if left == 0 else layer
        layer_r = max(0, layer - 1) if left == nqubits - 2 else layer
        tensor = qtn.Tensor(
            data=mat.reshape((2, 2, 2, 2)),
            inds=[
                f"{layer + 1}_{left}",
                f"{layer + 1}_{left + 1}",
                f"{layer_l}_{left}",
                f"{layer_r}_{left + 1}",
            ],
            left_inds=[
                f"{layer + 1}_{left}",
                f"{layer + 1}_{left + 1}",
            ],
        )
        tensor.apply_to_arrays(
            lambda x: torch.tensor(x, dtype=torch.complex128))
        self.tensor = tensor

    def layer(self) -> int:
        """
        Return the layer index of `self`.
        """
        try:
            (l, r) = self.tensor.left_inds
        except:
            (l, r) = self.tensor.inds[:2]
        dl = int(l.split("_")[0])
        dr = int(r.split("_")[0])
        assert dl == dr
        return dl - 1

    def qubits(self) -> Tuple[int, int]:
        """
        Return the left and right qubit indices on which `self` acts.
        """
        try:
            (l, r) = self.tensor.left_inds
        except:
            (l, r) = self.tensor.inds[:2]
        kl = int(l.split("_")[1])
        kr = int(r.split("_")[1])
        assert kr == kl + 1
        return (kl, kr)

def rand_gates(
    circuit: CircuitShape,
    seed: int,
) -> List[GateTensor]:
    """
    Return a list of tensors giving the gates of a Haar-random brickwork
    circuit, indexed according to the given `CircuitShape`. Each tensor in the
    list is given four indices such that the gate operating on the `q`-th and `p
    = q + 1`-th qubits in the `d`-th layer has indices following the form
    ('{d+1}_{q}', '{d+1}_{p}', '{d}_{q}', '{d}_{p}') where the first two are
    output indices and the last two are input indices, both ordered by qubit
    index.
    """
    unis = rand_unis_2q(seed)
    gates = list()
    for layer in range(circuit.depth):
        for k in range(circuit.gates_in_layer(layer)):
            mat = next(unis)
            left_idx = layer % 2 + 2 * k
            gate = GateTensor(circuit.nqubits, layer, left_idx, mat)
            gates.append(gate)
    return gates

class Circuit:
    """
    Thin wrapper around a list of `GateTensor`s along with the `CircuitShape`
    are indexed by.
    """
    seed: int
    shape: CircuitShape
    gates: List[GateTensor]

    @staticmethod
    def from_shape(shape: CircuitShape, seed: int) -> Self:
        """
        Construct from a set of randomly sampled gates that fill `shape`.
        """
        gates = rand_gates(shape, seed)
        self = Circuit()
        self.seed = seed
        self.shape = shape
        self.gates = gates
        return self

    def tiles(self, rank: int) -> List[List[GateTensor]]:
        """
        Like `CircuitShape.tiles`, but directly returning gates instead of
        indices.
        """
        return [
            [self.gates[k] for k in tile]
            for tile in self.shape.tiles(rank)
        ]

class MPS:
    """
    Newtype over a `quimb.tensor.MatrixProductState` to ensure a particular
    index labeling scheme.
    """
    mps: qtn.MatrixProductState

    def __init__(
        self,
        basis_state: str,
        ind_offs: int = 0,
        with_k: bool = True,
    ):
        """
        Construct a new `MPS` in a given computational basis state.
        `basis_state` should be a string comprising only '0's and '1's. Pass
        `with_k = False` to leave out leading 'k's from physical indices.
        """
        mps = qtn.MPS_computational_state(basis_state)
        for (k, tens) in enumerate(mps.tensors):
            k_offs = k + ind_offs
            ind = f"k0_{k_offs}" if with_k else f"0_{k_offs}"
            tens.modify(
                inds=[*tens.inds[:-1], ind], left_inds=[ind])
        mps.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.complex128))
        self.mps = mps

    @staticmethod
    def with_depths(
        basis_state: str,
        local_depth: List[int],
        ind_offs: int = 0,
        with_k: bool = True,
    ) -> Self:
        """
        Construct a new `MPS` in a given computational basis state, with indices
        encoding a "local depth" (i.e. how far each qubit has advanced in an
        overall circuit). `basis_state` should be a string comprising only '0's
        and '1's. The `k`-th element of `local_depth` should be the local depth
        of the `k`-th qubit, and should be of equal length as `basis_state`.
        Pass `with_k = False` to leave out leading 'k's from physical indices.
        """
        assert len(basis_state) == len(local_depth)
        mps = MPS(basis_state)
        for (k, (tens, depth)) in enumerate(zip(mps.mps.tensors, local_depth)):
            k_offs = k + ind_offs
            ind = f"k{depth}_{k_offs}" if with_k else f"{depth}_{k_offs}"
            tens.modify(inds=[*tens.inds[:-1], ind], left_inds=[ind])
        return mps

    def apply_gate(self, gate: GateTensor) -> None:
        """
        Apply a gate tensor to `self` in place. This function examines the index
        labels of the tensor object to figure out which qubits to apply the gate
        to, and to verify that `gate` is applied at the appropriate circuit
        depth.
        """
        new_inds = gate.tensor.inds[:2]
        pos = [int(ind.split("_")[1]) for ind in new_inds]
        self.mps = self.mps.gate_split(
            gate.tensor.data.reshape((4, 4)), where=gate.tensor.inds[-2:])
        for (k, new) in zip(pos, new_inds):
            tens = self.mps.tensors[k]
            new_ind = f"k{new}"
            tens.modify(inds=[*tens.inds[:-1], new_ind], left_inds=[new_ind])

    def _remove_k(self) -> None:
        """
        Remove a leading 'k' from all physical indices.
        """
        for tens in self.mps.tensors:
            ind = no_k(tens.inds[-1])
            tens.modify(inds=[*tens.inds[:-1], ind], left_inds=[ind])

    def _prepend_k(self) -> None:
        """
        Prepend a leading 'k' to all physical indices.
        """
        for tens in self.mps.tensors:
            ind = with_k(tens.inds[-1])
            tens.modify(inds=[*tens.inds[:-1], ind], left_inds=[ind])

    def inds(self) -> List[str]:
        return [tens.left_inds[0] for tens in self.mps.tensors]

def no_k(s: str) -> str:
    return s[1:] if s.startswith("k") else s

def with_k(s: str) -> str:
    return ("k" + s) if not s.startswith("k") else s

