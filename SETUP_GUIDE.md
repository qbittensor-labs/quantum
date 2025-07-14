# Subnet 63 Quantum Setup Guide

This guide will walk you through setting up and running as a miner or validator on the subnet.

## Prerequisites

- **Python 3.11** (required)
- **PM2** process manager (recommended)
- **Git** for cloning the repository and updating
- **CUDA-compatible GPU** (recommended for better performance)

## Setup Steps

### 1. Install Python 3.11 or above

Ensure you have Python 3.11 installed on your system. You can check your Python version with:
```bash
python3 --version
```

### 2. Create and Activate Virtual Environment

Create a new Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Bittensor

Install the latest version of Bittensor:
```bash
pip install --upgrade bittensor
pip install bittensor-cli # If you don't already have it
```

### 4. Register Your Wallet

You'll need to register your wallet on subnet 63:

```bash
# Register on subnet 63
btcli subnet register --wallet.name your_wallet_name --wallet.hotkey your_hotkey_name --netuid 63
```

### 5. Clone and Install the Repository

```bash
git clone https://github.com/qbittensor-labs/quantum.git
cd quantum
pip install -e .
```

### 6. Install PM2 (if not already installed)

```bash
npm install -g pm2
```

---

## Running a Miner

### Miner Setup

activate your venv 

Start your miner using PM2:

```bash
pm2 start --name your_process_name_here "python neurons/miner.py --wallet.name your_wallet_name --wallet.hotkey your_hotkey_name --netuid 63 --subtensor.network put_network_here --axon.port 8091 --logging.trace --difficulty <float>
```

Difficulty defaults to 0 if you do not pass it as an argument.

### Testnet Miner Setup

activate your venv 

For testing on the testnet:

```bash
pm2 start --name your_process_name_here "python neurons/miner.py --wallet.name your_wallet_name --wallet.hotkey your_hotkey_name --netuid 380 --subtensor.network test--axon.port 8091 --logging.trace --difficulty <float>
```

### Configuring Miner Difficulty

Miners can configure their requested difficulty level by editing the `DESIRED_DIFFICULTY` constant in:
```
neurons/miner.py
```

```python
DESIRED_DIFFICULTY: float = 0.0  # Change this value to your desired difficulty
```
The default miner will not be able to handle larger circuits than 32 qubits. It's up to you to develop novel and cutting edge approaches to running larger circuits!

**Difficulty Levels:**
- `0`: Default difficulty
- `1` and above: Progressively harder circuits

Miners are only able to increase their difficulty in increments. You may go from 0.0 to 0.7, and then up in increments of 0.4 once proving you can solve larger and larger circuits.

> **warning** The lowest difficulty already produces 31 qubit circuits. High difficulties will likely cause OOM errors, you must implement your own simulation stack to handle large qubit counts efficiently!

### Custom Solvers

We recommend that miners implement custom quantum circuit solvers to stay competitive. See the documentation:
- **Custom Solver Guide**: [qbittensor/miner/docs/CUSTOM_SOLVER.md](qbittensor/miner/docs/CUSTOM_SOLVER.md)
- **Custom Simulators**: [qbittensor/miner/docs/CUSTOM_SIMS_PROCESSORS.md](qbittensor/miner/docs/CUSTOM_SIMS_PROCESSORS.md)

To add a custom solver:
1. Create `qbittensor/miner/solvers/custom_peaked_solver.py`
2. Implement the `CustomSolver` class with a `solve()` method
3. Restart your miner - it will automatically detect and use your custom solver

---

## Running a Validator

### Hardware Requirements

**Recommended for Validators:**
- **Very strong GPU** (Likely an H200)
- High-end CPU and sufficient RAM for circuit generation

> **Note**: Validators require substantial computational resources to generate quantum circuits fast enough to serve all miners. If you cannot meet these requirements, consider delegating to our validator.

### Validator Setup

> **Note**: We have a hardcoded list of Whitelisted Validators to satisfy our "certificate distribution" methods. While this enhances the capabilities of our subnet, it adds friction in getting your validator up and running. Please reach out to us directly if you want your Validator added to the Whitelist.

activate your venv 

Start your validator using PM2:

```bash
pm2 start --name your_process_name_here "python neurons/validator.py --wallet.name your_wallet_name --wallet.hotkey your_hotkey_name --netuid 63 --subtensor.network put_network_here --logging.debug
```

### Testnet Validator Setup

activate your venv 

For testing on the testnet:

```bash
pm2 start --name your_process_name_here "python neurons/validator.py --wallet.name your_wallet_name --wallet.hotkey your_hotkey_name --netuid 380 --subtensor.network test --logging.debug

```

**Important**: Validator-miner collusion is strictly monitored and will result in removal from the whitelist.

---

## Troubleshooting

### Common Issues

1. **Port Already in Use**: Change the `--axon.port` parameter to a different port
3. **Connection Issues**: Check firewall settings and network connectivity
4. **Import Errors**: Ensure all dependencies are installed with `pip install -e .`

---

## Tips

### For Miners

1. **Use GPU acceleration** when possible
2. **Implement custom solvers** - cutting edge solutions able to handle large circuits will be rewarded more
3. **Monitor memory usage** - large circuits (>32 qubits) require significant resources or fancy tricks
4. **Optimize difficulty settings** based on your hardware capabilities

### For Validators

1. **Use high-end GPUs** for circuit generation
3. **Keep certificates secure** - they're required for miner verification
4. **Bugs** - Notify us when you encounter problems or bugs

---

For more detailed information about the quantum computing aspects and subnet mechanics, refer to the main [README.md](README.md). 