from math import ceil, log
import sys
import quimb as qu
import quimb.tensor as qtn
from lib.circuit_meta import *
from lib.optim import *

def main() -> None:
    nqubits = 36
    depth = nqubits // 2
    tile_width = ceil(log(nqubits))
    seed = 10546
    print(f"{nqubits = }")
    print(f"{depth = }")
    print(f"{tile_width = }")
    print("uniform prob =", 1 / 2 ** nqubits)
    circ = CircuitShape(nqubits, depth, tile_width).sample_gates(seed)
    print()

    # target_state = "".join("0" if k % 2 == 0 else "1" for k in range(nqubits))
    # optim_circuit(
    #     circ,
    #     target_state,
    #     pqc_prop=1.0,
    #     target_peak=1000.0,
    #     maxiters=5000,
    #     epsilon=1e-6,
    # )

    target_peak = 1000 / 2 ** nqubits
    print(f"{target_peak = }")
    (output_state, peak_prob_est) = optim_circuit_indep(
        circ,
        pqc_prop=2 / 3,
        target_peak=target_peak,
        maxiters=5000,
        epsilon=1e-6,
    )
    print(f"{output_state = }")
    print(f"{peak_prob_est = }")
    print(f"{peak_prob_est / target_peak = }")

if __name__ == "__main__":
    main()
