from math import ceil, log
import sys
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import quimb as qu
import quimb.tensor as qtn

from lib.circuit import *
from lib.circuit_meta import *
from lib.optim import *

def main() -> None:
    nqubits = 28
    depth = nqubits // 2
    tile_width = ceil(log(nqubits))
    # seed = 10546
    # seed = 10547
    seed = 10548
    # seed = 10549
    print(f"nqubits = {nqubits}")
    print(f"depth = {depth}")
    print(f"tile width = {tile_width}")
    print(f"uniform prob = {1 / 2 ** nqubits:g}")
    circ = CircuitShape(nqubits, depth, tile_width).sample_gates(seed)

    # target_peak = 10000 / 2 ** nqubits
    target_peak = 5000 / 2 ** nqubits
    # target_peak = 1.0
    print(f"target peak = {target_peak:g}")
    print()

    t0 = time.time()
    peaked = PeakedCircuit.from_circuit(
        circ,
        target_peak,
        # pqc_prop=2 / 3,
        pqc_prop=1.0,
        maxiters=5000,
        epsilon=1e-6,
    )
    qasm = peaked.to_qasm()
    print()
    print(f"circuit gen finished in {time.time() - t0:g} seconds")

    print(f"target state = {peaked.target_state}")
    print(f"estimated peak probability {peaked.peak_prob_est:g}")
    print(f"estimated peak / target peak = {peaked.peak_prob_est / target_peak:g}")
    print()

    if nqubits > 28:
        print("too many qubits, skipping solution")
        sys.exit(0)

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
    print(f"solution finished in {time.time() - t0:g} seconds")
    correct = solution == peaked.target_state
    print(f"SOLUTION:")
    print(f"peak state = {solution}")
    print(f"peak probability = {peak_prob:g}")
    print(f"correct = {correct}")
    if not correct:
        print("  target = ", end="")
        for (sol, targ) in zip(solution, peaked.target_state):
            if sol == targ:
                print("_", end="")
            else:
                print(targ, end="")
        print()
        print("solution = ", end="")
        for (sol, targ) in zip(solution, peaked.target_state):
            if sol == targ:
                print("_", end="")
            else:
                print(sol, end="")
        print()
    print()

if __name__ == "__main__":
    main()
