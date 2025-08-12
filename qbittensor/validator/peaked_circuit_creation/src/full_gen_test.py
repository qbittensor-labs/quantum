from math import ceil, log
import sys
import quimb as qu
import quimb.tensor as qtn
from lib.circuit import *
from lib.circuit_meta import *
from lib.optim import *

def main() -> None:
    nqubits = 20
    depth = nqubits // 2
    tile_width = ceil(log(nqubits))
    seed = 10546
    print(f"{nqubits = }")
    print(f"{depth = }")
    print(f"{tile_width = }")
    print("uniform prob =", 1 / 2 ** nqubits)
    circ = CircuitShape(nqubits, depth, tile_width).sample_gates(seed)

    target_peak = 1000 / 2 ** nqubits
    print(f"{target_peak = }")
    peaked = PeakedCircuit.from_circuit(
        circ,
        target_peak,
        pqc_prop=2 / 3,
        maxiters=5000,
        epsilon=1e-6,
    )
    print(f"target state = {peaked.target_state}")
    print(f"estimated peak probability {peaked.peak_prob_est}")
    print(f"est. peak / target peak = {peaked.peak_prob_est / target_peak}")

    qasm = peaked.to_qasm()
    print(f"qasm length = {len(qasm)}")
    with open("peaked_test.qasm", "w") as outfile:
        outfile.write(qasm)

if __name__ == "__main__":
    main()
