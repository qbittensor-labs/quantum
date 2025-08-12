import quimb as qu
import quimb.tensor as qtn
from lib.circuit_meta import *

def main() -> None:
    circ = CircuitShape(6, 1, 3)
    gates = rand_gates(circ, 10546)
    for g in gates:
        print(g)
    print()
    mps = qtn.MPS_computational_state(circ.nqubits * "0")
    for (k, tens) in enumerate(mps.tensors):
        tens.modify(inds=[*tens.inds[:-1], f"k0_{k}"], left_inds=[f"k0_{k}"])
        print(tens)
    print()
    for g in gates:
        mps.gate_split(g.data.reshape((4, 4)), where=g.inds[-2:])
    print()
    for tens in mps.tensors:
        print(tens)

if __name__ == "__main__":
    main()
