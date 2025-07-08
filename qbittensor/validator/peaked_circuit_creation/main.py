import torch
from lib.circuit_gen import DEVICE, CircuitParams


def main():
    # This is for debug
    if DEVICE == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    difficulty_level = 1.0
    seed = 1054
    circuit_params = CircuitParams.from_difficulty(difficulty_level)
    print(f"qubits: {circuit_params.nqubits}")
    print(f"rqc depth: {circuit_params.rqc_depth}")
    print(f"pqc depth: {circuit_params.pqc_depth}")
    circuit = circuit_params.compute_circuit(seed)
    qasm_out = circuit.to_qasm()
    print(qasm_out)
    print("Target state")
    print(circuit.target_state)


if __name__ == "__main__":
    main()


# 0.0: 001101001111
# 1.0: 0011010011110000011111
# with gpu: 1010110011110100001011, seed 1054, diff 1.0
