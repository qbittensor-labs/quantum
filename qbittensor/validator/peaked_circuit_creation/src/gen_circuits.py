from dataclasses import dataclass
import json
from math import ceil, log, sqrt
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Tuple
from zipfile import ZipFile

import numpy as np

from lib.circuit import *
from lib.circuit_meta import *
from lib.optim import *

@dataclass
class Meta:
    seed: int
    nqubits: int
    target_prob: float
    gen_time: float
    target: str
    peak_prob_est: float
    file: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": int(self.seed),
            "nqubits": int(self.nqubits),
            "target_prob": float(self.target_prob),
            "gen_time": float(self.gen_time),
            "target": str(self.target),
            "peak_prob_est": float(self.peak_prob_est),
            "file": self.file,
        }

GEN_COUNT = 0
GEN_TOTAL = None

def gen(nqubits: int, target_peaking: float, seed: int) -> Tuple[str, str]:
    global GEN_COUNT, GEN_TOTAL
    GEN_COUNT += 1
    print(f"====================")
    print(f"{GEN_COUNT} / {GEN_TOTAL}")
    print(f"nqubits: {nqubits}")
    print(f"target peaking: {target_peaking:g}")
    print(f"seed: {seed}")
    print(f"====================")
    depth = nqubits // 2
    # tile_width = ceil(log(nqubits))
    tile_width = ceil(sqrt(nqubits))
    circuit = CircuitShape(nqubits, depth, tile_width).sample_gates(seed)
    target_prob = min(1, target_peaking / 2 ** nqubits)

    t0 = time.time()
    peaked = PeakedCircuit.from_circuit(
        circuit,
        target_prob,
        pqc_prop=1.0,
        maxiters=5000,
        epsilon=1e-6,
    )
    qasm = peaked.to_qasm()
    gen_time = time.time() - t0
    print(f"Gen finished in {gen_time:g} seconds")

    fname = f"peaked2-0_{nqubits=}_{seed=}.qasm"
    with open(fname, "w") as outfile:
        outfile.write(qasm)

    meta = Meta(
        seed,
        nqubits,
        target_prob,
        gen_time,
        peaked.target_state,
        peaked.peak_prob_est,
        fname,
    )
    fname_meta = f"peaked2-0_{nqubits=}_{seed=}_meta.json"
    with open(fname_meta, "w") as outfile:
        json.dump(meta.to_dict(), outfile)

    return (fname, fname_meta)

def do_gens(ntries: int, nqubits: int) -> List[Tuple[str, str]]:
    a = 0.22582936781580765
    b = 0.37283850723802836
    target_peaking = 10 ** (a * nqubits + b + 0.5)
    return [
        gen(nqubits, target_peaking, int(2 ** 16 * random.random()))
        for _ in range(ntries)
    ]

def main() -> None:
    mc = 10
    nqubits = list(range(30, 41, 2))
    # nqubits = [30]

    global GEN_TOTAL
    GEN_TOTAL = mc * len(nqubits)

    files = [x for nq in nqubits for x in do_gens(mc, nq)]
    timestamp = time.strftime("%Y%m%d-%H%M", time.localtime())
    zipfname = f"peaked2-0_{timestamp}.zip"
    with ZipFile(zipfname, mode="w") as zipfile:
        for (fname, fname_meta) in files:
            zipfile.write(fname)
            zipfile.write(fname_meta)
            os.remove(fname)
            os.remove(fname_meta)

if __name__ == "__main__":
    main()
