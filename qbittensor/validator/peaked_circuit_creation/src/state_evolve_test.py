import quimb as qu
import quimb.tensor as qtn
from lib.circuit_meta import *

def main() -> None:
    circ = CircuitShape(6, 2, 3)
    gates = rand_gates(circ, 10546)
    mps = MPS(circ.nqubits * "0")
    for tens in mps.mps.tensors:
        print(tens.inds, tens.left_inds)
    print()
    for g in gates:
        print(g.tensor.inds, g.tensor.left_inds)
        mps.apply_gate(g)
    print()
    for tens in mps.mps.tensors:
        print(tens.inds, tens.left_inds)

if __name__ == "__main__":
    main()
