# Custom Quantum Circuit Solver Guide

## Overview

This subnet supports custom circuit solvers through a simple template pattern. Miners can implement their own solving algorithms by creating a standardized custom solver file in the `solvers` directory.

## EXACT REQUIREMENTS

**Your custom solver MUST follow these exact specifications:**

1. **File location**: `miner/solvers/custom_peaked_solver.py` (in the solvers directory)
2. **File name**: `custom_peaked_solver.py` (exactly this name)
3. **Class name**: `CustomSolver` (exactly this name)
4. **Method name**: `solve` (exactly this name)
5. **Method signature**: `def solve(self, qasm: str) -> str:`

**Example:**
```python
# File: miner/solvers/custom_peaked_solver.py
class CustomSolver:
    def __init__(self):
        # Your initialization here
        pass

    def solve(self, qasm: str) -> str:
        # Your implementation here
        return "peak_bitstring_or_empty_string"
```

**The code will only detect your solver if you use these exact names and location!**

---

## Quick Start

1. **Navigate to your miner directory**:
   ```bash
   cd your_miner_directory/qbittensor/miner/solvers/
   ```

2. **Copy the template** (see below) to `custom_peaked_solver.py`

3. **Implement your solver logic** in the `solve()` method

4. **Test your solver** with the provided test examples

5. **Restart your miner** - it will automatically detect and use your custom solver

---

## Input/Output Specification

### Input Format
- **Type**: `str` (OpenQASM 2.0 format)
- **Example**:
```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[31];
creg c[31];
h q[0];
cx q[0],q[1];
// ... more gates ...
measure q -> c;
```

### Output Format
- **Success**: Peak bitstring as string (e.g., `"110010101"`)
- **Failure**: Empty string (`""`)
- **Important**: Return `""` for any error (OOM, timeout, etc.)

---

## Complete Template

Copy this template to get started:

```python
# File: miner/solvers/custom_peaked_solver.py
"""
Custom Quantum Circuit Solver Template

Replace this template with your own implementation.
The solver MUST implement the exact interface specified.
"""

import bittensor as bt
import time
import traceback
from pathlib import Path

class CustomSolver:
    """
    Custom quantum circuit solver implementation.

    This class will be automatically detected and used by the miner
    if placed in: miner/solvers/custom_peaked_solver.py
    """

    def __init__(self):
        """
        Initialize your custom solver.

        Configure your solver here - load tooling, set up backends,
        initialize any resources you need.
        """
        # Your initialization code here
        # Examples:
        # self.backend = MyQuantumBackend()
        # self.config = load_config()
        # self.pipeline = load_my_pipeline()

        bt.logging.info("CustomSolver initialized")

    def solve(self, qasm: str) -> str:
        """
        Solve a quantum circuit to find the peak bitstring.

        Args:
            qasm: OpenQASM 2.0 string representation of the circuit

        Returns:
            str: Peak bitstring (e.g., "110010") or empty string if failed
        """
        try:
            # IMPLEMENT YOUR SOLVER HERE
            # Replace this with your custom algorithm:

            # Option A: Use default pipeline with modifications
            # from qbittensor.miner.solvers.default_peaked_solver import solve_peaked_circuit
            # result = solve_peaked_circuit(qasm)
            # return result.get("peak_bitstring", "")

            # Option B: Completely custom implementation
            # peak_bitstring = my_custom_algorithm(qasm)
            # return peak_bitstring

            # Option C: Classical-quantum hybrid approach
            # if should_use_classical_method(qasm):
            #     return classical_solver(qasm)
            # else:
            #     return quantum_solver(qasm)

            # For template demo - REPLACE THIS WITH YOUR SOLVER:
            nqubits = self._count_qubits(qasm)
            return "0" * nqubits

        except MemoryError as e:
            self._log_oom_error(qasm, str(e))
            return ""
        except Exception as e:
            self._log_error(f"Solver failed: {e}")
            return ""

    # Helper methods (customize as needed)
    def _count_qubits(self, qasm: str) -> int:
        """Extract number of qubits from QASM string."""
        import re
        for line in qasm.split("\n"):
            if line.strip().startswith("qreg"):
                match = re.search(r"qreg\s+\w+\[(\d+)\]", line)
                if match:
                    return int(match.group(1))
        return 0

    def _log_oom_error(self, qasm: str, error_msg: str):
        """Log OOM errors to peaked-miner-err.log"""
        log_path = Path("peaked-miner-err.log")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"{timestamp} OOM_ERROR error='{error_msg}'\n")
        bt.logging.error(f"OOM error: {error_msg}")

    def _log_error(self, message: str):
        """Log general errors"""
        log_path = Path("peaked-miner-err.log")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"{timestamp} ERROR {message}\n")
        bt.logging.error(message)
```

---

## Available Components

You can reuse existing components in your custom solver:

### Default Solver Pipeline
```python
from qbittensor.miner.solvers.default_peaked_solver import solve_peaked_circuit

# Use the full default pipeline
result = solve_peaked_circuit(qasm)
peak_bitstring = result.get("peak_bitstring", "")
```

### Simulators
```python
from qbittensor.miner.simulator import create_simulator

sim = create_simulator("qiskit")
counts = sim.run(qasm, shots=1024)
```

### Task Processors
```python
from qbittensor.miner.task_processors import PeakedCircuitProcessor

processor = PeakedCircuitProcessor()
result = processor.process(counts)
peak_bitstring = result["peak_bitstring"]
```

---

## Error Handling

Your solver MUST handle errors gracefully:

1. **Always return `""` on any failure**
2. **Log OOM errors to `peaked-miner-err.log`**
3. **Don't let exceptions crash the miner process**
4. **Handle timeouts and resource limits**

---

## Verification
Check your miner logs for:
```
CustomSolver initialized
```

## Testing
```python
# File: test_my_solver.py (run from miner directory)
from solvers.custom_peaked_solver import CustomSolver

# Bell state test circuit
test_qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"""

solver = CustomSolver()
result = solver.solve(test_qasm)
print(f"Result: {result}")
```

**Run the test:**
```bash
cd your_miner_directory/qbittensor/miner/
python test_my_solver.py
```

---

## Integration Details

### How Detection Works

1. **Automatic Detection**: The miner checks for `solvers/custom_peaked_solver.py` on startup
2. **Fallback**: If no custom solver exists, uses the default solver
3. **Error Handling**: If custom solver fails, logs to `peaked-miner-err.log`

### Tips and Considerations

- **Memory**: Smart solutions are needed to run large circuits (>32 qubits)
- **Error Handling**: Always handle exceptions and return empty string on failure
- **Logging**: You can use the provided logging setup for debugging

---

## FAQ

**Q: What if my solver needs additional dependencies?**

A: Install them in your miner environment. The subnet will use your custom solver if detected.

**Q: How can I implement a custom simulator or processor?**

A. Extend the base simulator and processor classes. Reference the README.MD for more information.

**Q: Can I modify the default solver?**

A: You can modify `default_peaked_solver.py` but it is highly recommended that you create your own `custom_peaked_solver.py` instead.

**Q: How do I debug my solver?**

A: Configure write out to logs such as `peaked-miner-err.log` for error messages. Add print statements or additional logging.

**Q: What happens if my solver crashes?**

A: The miner will catch the exception, log it, and return an empty string (indicating failure).

---
