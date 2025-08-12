from dataclasses import dataclass
import json
from math import ceil, log, sqrt
import random
import sys
import time
from typing import Any, Dict, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import quimb as qu
import quimb.tensor as qtn

from lib.circuit import *
from lib.circuit_meta import *
from lib.optim import *

@dataclass
class Data:
    seed: int
    nqubits: int
    target_prob: float
    gen_time: float
    target: str
    peak_prob_est: float
    solve_time: float
    solution: str
    peak_prob: float
    correct: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": int(self.seed),
            "nqubits": int(self.nqubits),
            "target_prob": float(self.target_prob),
            "gen_time": float(self.gen_time),
            "target": str(self.target),
            "peak_prob_est": float(self.peak_prob_est),
            "solve_time": float(self.solve_time),
            "solution": str(self.solution),
            "peak_prob": float(self.peak_prob),
            "correct": bool(self.correct),
        }

GEN_SOLVE_COUNT = 0
GEN_SOLVE_TOTAL = None

def gen_and_solve(nqubits: int, target_peaking: float, seed: int) -> Data:
    global GEN_SOLVE_COUNT, GEN_SOLVE_TOTAL
    GEN_SOLVE_COUNT += 1
    print(f"====================")
    print(f"{GEN_SOLVE_COUNT} / {GEN_SOLVE_TOTAL}")
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

    t0 = time.time()
    circuit = (
        QuantumCircuit.from_qasm_str(qasm)
        .remove_final_measurements(inplace=False)
    )
    circuit.save_statevector()
    backend = AerSimulator(method="statevector")
    job = backend.run(circuit, shots=1)
    result = job.result()
    statevector = result.data(0)["statevector"]
    sv = np.array(statevector)
    peak_idx = np.argmax(np.abs(sv) ** 2)
    peak_prob = np.abs(sv[peak_idx]) ** 2
    solution = f"{peak_idx:0{nqubits}b}"[::-1]
    solve_time = time.time() - t0
    print(f"Solve finished in {solve_time:g} seconds")

    return Data(
        seed,
        nqubits,
        target_prob,
        gen_time,
        peaked.target_state,
        peaked.peak_prob_est,
        solve_time,
        solution,
        peak_prob,
        solution == peaked.target_state,
    )

def try_solves(ntries: int, nqubits: int, target_peaking: float) -> List[Data]:
    return [
        gen_and_solve(nqubits, target_peaking, int(2 ** 16 * random.random()))
        for _ in range(ntries)
    ]

def main() -> None:
    mc = 100
    # nqubits = list(range(18, 29, 2))
    nqubits = list(range(22, 29, 2))
    # peaking = [
    #     100.0,
    #     200.0,
    #     500.0,
    #     1000.0,
    #     2000.0,
    #     5000.0,
    #     10000.0,
    #     20000.0,
    #     50000.0,
    #     100000.0,
    #     200000.0,
    #     500000.0,
    # ]
    # peaking = [float(x) for x in np.logspace(np.log10(5e3), 7, 16)]
    peaking = [float(x) for x in np.logspace(5, 9, 16)]

    global GEN_SOLVE_TOTAL
    GEN_SOLVE_TOTAL = mc * len(nqubits) * len(peaking)

    data = {
        "mc": mc,
        "nqubits": nqubits,
        "peaking": peaking,
        "data": [
            [
                [d.to_dict() for d in try_solves(mc, nq, pk)]
                for pk in peaking
            ]
            for nq in nqubits
        ],
    }
    timestamp = time.strftime("%Y%m%d-%H%M", time.localtime())
    with open(f"outdata_{timestamp}.json", "w") as outfile:
        json.dump(data, outfile)

if __name__ == "__main__":
    main()
