# Custom Quantum Circuit Solver Guide

## Overview

This subnet supports custom circuit solvers through a simple template pattern. Miners can implement their own solving algorithms by creating a standardized custom solver file in the `solvers` directory.

## EXACT REQUIREMENTS

**Your custom solvers MUST follow these exact specifications:**

### For Peaked Circuit Solver:
1. **File location**: `miner/solvers/custom_peaked_solver.py` (in the solvers directory)
2. **File name**: `custom_peaked_solver.py` (exactly this name)
3. **Class name**: `CustomPeakedSolver` (exactly this name)
4. **Method name**: `solve` (exactly this name)
5. **Method signature**: `def solve(self, qasm: str) -> str:`

### For Hidden Stabilizer Circuit Solver:
1. **File location**: `miner/solvers/custom_hstab_solver.py` (in the solvers directory)  
2. **File name**: `custom_hstab_solver.py` (exactly this name)
3. **Class name**: `CustomHStabSolver` (exactly this name)
4. **Method name**: `solve` (exactly this name)
5. **Method signature**: `def solve(self, qasm: str) -> str:`

### Solver Combination Options:
- **No custom solvers**: Uses default solvers for both circuit types
- **Peaked only**: Create `custom_peaked_solver.py`, uses default for hstab
- **HStab only**: Create `custom_hstab_solver.py`, uses default for peaked
- **Both custom**: Create both files for full customization

**Examples:**
```python
# File: miner/solvers/custom_peaked_solver.py
class CustomPeakedSolver:
    def __init__(self):
        # Your initialization here
        pass

    def solve(self, qasm: str) -> str:
        # Your implementation here
        return "peak_bitstring_or_empty_string"
```

```python
# File: miner/solvers/custom_hstab_solver.py
class CustomHStabSolver:
    def __init__(self):
        # Your initialization here
        pass

    def solve(self, qasm: str) -> str:
        # Your implementation here
        return "concatenated_stabilizer_string_or_empty_string"
```

**The code will only detect your solvers if you use these exact names and locations!**

---

## Quick Start

1. **Navigate to your miner directory**:
   ```bash
   cd your_miner_directory/qbittensor/miner/solvers/
   ```

2. **Choose your customization level**:
   - For peaked circuits only: Copy peaked template to `custom_peaked_solver.py`
   - For hstab circuits only: Copy hstab template to `custom_hstab_solver.py`
   - For both: Create both files

3. **Implement your solver logic** in the `solve()` method(s)

4. **Test your solver(s)** with the provided test examples

5. **Restart your miner** - it will automatically detect and use your custom solver(s)

---

## Input/Output Specification

### Input Format (Both Circuit Types)
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

#### Peaked Circuits
- **Success**: Peak bitstring as string (e.g., `"110010101"`)
- **Failure**: Empty string (`""`)

#### HStab Circuits  
- **Success**: Concatenated stabilizer string (e.g., `"-XZYZ+ZZZZ+IXYZ-ZZZI"`)
- **Failure**: Empty string (`""`)

**Important**: Always return `""` for any error (OOM, timeout, etc.)

---

## Complete Templates

### Peaked Circuit Solver Template

Copy this template to `custom_peaked_solver.py`:

```python
# File: miner/solvers/custom_peaked_solver.py
"""
Custom Peaked Circuit Solver Template

Replace this template with your own implementation.
The solver MUST implement the exact interface specified.
"""

import bittensor as bt
import time
import traceback
from pathlib import Path

class CustomPeakedSolver:
    """
    Custom peaked circuit solver implementation.

    This class will be automatically detected and used by the miner
    if placed in: miner/solvers/custom_peaked_solver.py
    """

    def __init__(self):
        """
        Initialize your custom peaked solver.

        Configure your solver here - load tooling, set up backends,
        initialize any resources you need for peaked circuits.
        """
        # Your initialization code here
        # Examples:
        # self.backend = MyQuantumBackend()
        # self.config = load_config()
        # self.pipeline = load_my_pipeline()

        bt.logging.info("CustomPeakedSolver initialized")

    def solve(self, qasm: str) -> str:
        """
        Solve a peaked circuit to find the peak bitstring.

        Args:
            qasm: OpenQASM 2.0 string representation of the peaked circuit

        Returns:
            str: Peak bitstring (e.g., "110010") or empty string if failed
        """
        try:
            # IMPLEMENT YOUR PEAKED SOLVER HERE
            # Replace this with your custom algorithm:

            # Option A: Use default pipeline with modifications
            # from qbittensor.miner.solvers.default_peaked_solver import DefaultPeakedSolver 
            # default_solver = DefaultPeakedSolver()
            # result = default_solver.solve(qasm)
            # return result

            # Option B: Completely custom implementation
            # peak_bitstring = my_custom_peaked_algorithm(qasm)
            # return peak_bitstring

            # For template demo - REPLACE THIS WITH YOUR SOLVER:
            nqubits = self._count_qubits(qasm)
            return "0" * nqubits

        except MemoryError as e:
            self._log_oom_error(qasm, str(e), "peaked")
            return ""
        except Exception as e:
            self._log_error(f"Peaked solver failed: {e}")
            return ""

    def _count_qubits(self, qasm: str) -> int:
        """Extract number of qubits from QASM string."""
        import re
        for line in qasm.split("\n"):
            if line.strip().startswith("qreg"):
                match = re.search(r"qreg\s+\w+\[(\d+)\]", line)
                if match:
                    return int(match.group(1))
        return 0

    def _log_oom_error(self, qasm: str, error_msg: str, circuit_type: str):
        """Log OOM errors to peaked-miner-err.log"""
        log_path = Path("peaked-miner-err.log")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"{timestamp} OOM_ERROR type={circuit_type} error='{error_msg}'\n")
        bt.logging.error(f"OOM error in {circuit_type}: {error_msg}")

    def _log_error(self, message: str):
        """Log general errors"""
        log_path = Path("peaked-miner-err.log")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"{timestamp} ERROR {message}\n")
        bt.logging.error(message)
```

### Hidden Stabilizer Circuit Solver Template

Copy this template to `custom_hstab_solver.py`:

```python
# File: miner/solvers/custom_hstab_solver.py
"""
Custom Hidden Stabilizer Circuit Solver Template

Replace this template with your own implementation.
The solver MUST implement the exact interface specified.
"""

import bittensor as bt
import time
import traceback
from pathlib import Path

class CustomHStabSolver:
    """
    Custom hidden stabilizer circuit solver implementation.

    This class will be automatically detected and used by the miner
    if placed in: miner/solvers/custom_hstab_solver.py
    """

    def __init__(self):
        """
        Initialize your custom hstab solver.

        Configure your solver here - load tooling, set up backends,
        initialize any resources you need for hstab circuits.
        """
        # Your initialization code here
        # Examples:
        # self.backend = MyQuantumBackend()
        # self.config = load_config()
        # self.pipeline = load_my_hstab_pipeline()

        bt.logging.info("CustomHStabSolver initialized")

    def solve(self, qasm: str) -> str:
        """
        Solve a hidden stabilizer circuit to find stabilizer generators.

        Args:
            qasm: OpenQASM 2.0 string representation of the hstab circuit

        Returns:
            str: Concatenated stabilizer string (length n×n) or empty string if failed
        """
        try:
            # IMPLEMENT YOUR HSTAB SOLVER HERE
            # Replace this with your custom algorithm:

            # Option A: Use default pipeline with modifications
            # from qbittensor.miner.solvers.default_hstab_solver import DefaultHStabSolver
            # default_solver = DefaultHStabSolver()
            # result = default_solver.solve(qasm)
            # return result

            # Option B: Completely custom implementation
            # stabilizer_string = my_custom_hstab_algorithm(qasm)
            # return stabilizer_string

            # For template demo - REPLACE THIS WITH YOUR SOLVER:
            nqubits = self._count_qubits(qasm)
            return "I" * (nqubits * nqubits)  # Placeholder

        except MemoryError as e:
            self._log_oom_error(qasm, str(e), "hstab")
            return ""
        except Exception as e:
            self._log_error(f"HStab solver failed: {e}")
            return ""

    def _count_qubits(self, qasm: str) -> int:
        """Extract number of qubits from QASM string."""
        import re
        for line in qasm.split("\n"):
            if line.strip().startswith("qreg"):
                match = re.search(r"qreg\s+\w+\[(\d+)\]", line)
                if match:
                    return int(match.group(1))
        return 0

    def _log_oom_error(self, qasm: str, error_msg: str, circuit_type: str):
        """Log OOM errors to hstab-miner-err.log"""
        log_path = Path("hstab-miner-err.log")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"{timestamp} OOM_ERROR type={circuit_type} error='{error_msg}'\n")
        bt.logging.error(f"OOM error in {circuit_type}: {error_msg}")

    def _log_error(self, message: str):
        """Log general errors"""
        log_path = Path("hstab-miner-err.log")
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
2. **Log OOM errors to a log if needed: `peaked-miner-err.log`**
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

1. **Automatic Detection**: The miner checks for `solvers/custom_{circuit_type}_solver.py` on startup
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

A: You can modify the default solvers but it is highly recommended that you create your own `custom_peaked_solver.py` or `custom_hstab_solver.py`instead.

**Q: How do I debug my solver?**

A: Configure write out to logs such as `peaked-miner-err.log` for error messages. Add print statements or additional logging.

**Q: What happens if my solver crashes?**

A: The miner will catch the exception, log it, and return an empty string (indicating failure).

---
