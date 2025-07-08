# Custom Simulators and Processors Guide

### 1. Simulator Backend (`simulator/`)

The core miner circuit simulation layer:

```python
from simulator import create_simulator

sim = create_simulator('qiskit')
counts = sim.run(qasm_string, shots=1024)
```

**Files:**
- `base.py` - Abstract `QuantumSimulator` interface
- `default_sim.py` - Default Qiskit implementation
- `__init__.py` - Factory function for creating simulators

### 2. Circuit Solver Architecture (`solvers/`)

The miner automatically detects and uses custom solvers, or falls back to the default solver:

```python
from qbittensor.miner.services.circuit_solver import CircuitSolver

# CircuitSolver automatically:
# 1. Checks for custom_peaked_solver.py
# 2. Falls back to DefaultPeakedSolver if no custom solver
solver = CircuitSolver(base_dir="path/to/base")
```

**Default Solver Strategy:**
- **≤32 qubits**: GPU-accelerated statevector (if available), CPU fallback
- **>32 qubits (Will likely throw OOM)**: MPS method (not ideal), Recommended to implement your own custom simulation stack for >32 qubits.

**For custom solvers, see:** `solvers/CUSTOM_SOLVER.md`

### 3. Task Processors (`task_processors/`)

Task-specific result processing:

```python
from task_processors import PeakedCircuitProcessor

# For exact statevector processing
processor = PeakedCircuitProcessor(use_exact=True)
result = processor.process(statevector)

# For sampling/counts processing
processor = PeakedCircuitProcessor(use_exact=False)
result = processor.process(counts)

# Returns: {'peak_bitstring': '0011', 'peak_probability': 0.85, 'peaking_ratio': 15.2}
```

**Files:**
- `base.py` - Abstract `TaskProcessor` interface
- `default_peaked_processor.py` - Processor for peaked circuit challenges

### 4. Basic Integration Flow

```python
from simulator import create_simulator
from task_processors import PeakedCircuitProcessor

sim = create_simulator('qiskit', method='statevector', device='GPU')

statevector = sim.get_statevector(qasm_string)

processor = PeakedCircuitProcessor(use_exact=True)

result = processor.process(statevector)

peak_bitstring = result['peak_bitstring']
```

### Adding Custom Simulator Backend

```python
#simulator/my_custom_simulator.py
from simulator.base import QuantumSimulator
import numpy as np

class MyCustomSimulator(QuantumSimulator):
    def __init__(self, **kwargs):
        # Add your inits here
        self.device = kwargs.get('device', 'CPU')

    def run(self, qasm: str, shots: int = 1024) -> Dict[str, int]:
        # Must return: {"bitstring": count, "bitstring": count, ...}
        # Example: {"00": 512, "01": 256, "10": 128, "11": 128}
        return self._my_simulation_method(qasm, shots)

    def _my_simulation_method(self, qasm: str, shots: int) -> Dict[str, int]:
        """
        Your custom simulation implementation goes here.
        """
        # Dummy example implementation, replace with your method:
        import random

        num_qubits = self._count_qubits(qasm)
        possible_outcomes = [format(i, f'0{num_qubits}b') for i in range(2**num_qubits)]

        counts = {}
        for _ in range(shots):
            outcome = random.choice(possible_outcomes)
            counts[outcome] = counts.get(outcome, 0) + 1

        return counts

    def _count_qubits(self, qasm: str) -> int:
        """Helper to extract qubit count from QASM."""
        import re
        for line in qasm.split("\n"):
            if line.strip().startswith("qreg"):
                match = re.search(r"qreg\s+\w+\[(\d+)\]", line)
                if match:
                    return int(match.group(1))
        return 1

    def get_info(self) -> Dict[str, Any]:
        return {"backend": "my_custom", "device": self.device}

# Integration option 1: Extend the factory
def create_simulator(backend='qiskit', **kwargs):
    if backend == 'my_custom':
        return MyCustomSimulator(**kwargs)
    elif backend == 'qiskit':
        return DefaultSim(**kwargs)

# Integration option 2: Direct instantiation
from custom_backends.my_simulator import MyCustomSimulator
custom_sim = MyCustomSimulator()

# Usage with processors
from task_processors import PeakedCircuitProcessor

counts = custom_sim.run(qasm_string, shots=1024)
processor = PeakedCircuitProcessor(use_exact=False)
result = processor.process(counts)
```

### Adding Custom Task Processors

```python
# task_processors/my_processor.py
from task_processors.base import TaskProcessor

class MyCustomProcessor(TaskProcessor):
    def process(self, data, **kwargs):
        # Add your processing logic here

        # Example: find most frequent bitstring
        if isinstance(data, dict):
            peak_bitstring = max(data.keys(), key=lambda x: data[x])
        else:
            probabilities = np.abs(data) ** 2
            peak_idx = np.argmax(probabilities)
            peak_bitstring = format(peak_idx, f"0{int(np.log2(len(data)))}b")[::-1]

        return {
            "peak_bitstring": peak_bitstring,
            "custom_metric": self.calculate_custom_metric(data)
        }

    def validate_result(self, result):
        return result.get("peak_bitstring") is not None

processor = MyCustomProcessor()
result = processor.process(counts_or_statevector)
```

## Configuration

Example simulator configuration:

```python
# Default
sim = create_simulator(
    backend='qiskit',
    method='statevector',
    device='GPU',
    precision='double'
)
# Or load your custom simulator
custom_sim = create_simulator(
    backend='my_custom',
    use_approximation=True,
    device='GPU'
)
```
