from .base import QuantumSimulator
from .default_sim import DefaultSim


def create_simulator(backend: str = "qiskit", **kwargs) -> QuantumSimulator:
    """
    Create a quantum simulator instance

    Args:
        backend: Backend name (default: 'qiskit')
        **kwargs: Backend-specific configuration

    Returns:
        QuantumSimulator instance

    Example:
        >>> sim = create_simulator('qiskit', method='statevector')
        >>> counts = sim.run(qasm_string, shots=1024)
    """
    if backend == "qiskit":
        return DefaultSim(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Available backends: ['qiskit']")


__all__ = ["create_simulator", "QuantumSimulator", "DefaultSim"]
