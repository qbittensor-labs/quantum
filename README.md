<div align="center">

## qBitTensor

</div>


<img src="./img/quantum-subnet.png" />

# qBitTensor: Unlocking Practical Quantum Computing on Bittensor

Welcome to the Bittensor **Quantum Subnet**, an initiative engineered to accelerate the journey of quantum computing from the lab to practical, real-world application. We're building a decentralized network where innovation thrives, access is democratized, and some of the near-term challenges of quantum technology are systematically addressed through collaborative effort.

## Our Core Purpose
qBitTensor is fundamentally about advancing Quantum Computing beyond theoretical research to practical application. We believe the next breakthroughs will emerge from a nexus of diverse perspectives and shared computational power. Our mission is multifaceted:

1. **Democratize Quantum Access:** Break down the barriers to entry, making quantum computation accessible to a wider audience, from researchers to startups.

2. **Drive Innovation:** Foster a vibrant ecosystem where novel approaches to quantum problem-solving are incentivized, discovered, and refined.

3. **Leverage Decentralization:** Utilize Bittensor's unique incentivized network structure to explore entirely new paradigms for distributed quantum problem-solving and resource optimization.

We're not just theorizing; we're building the infrastructure for a future where quantum power is a shared utility, not a privileged resource.

## Initial Problem Focus: Peaked Circuits

To kick things off, our initial focus is on the execution of "Peaked Circuits" to find a hidden bitstring. These circuits, explored in depth by [Scott Aaronson](https://scholar.google.com/citations?user=EYv2BNQAAAAJ&hl=en) and [Yuxuan Zhang](https://scholar.google.com/citations?user=jf5oJUUAAAAJ&hl=en&oi=sra) in their paper "[On verifiable quantum advantage with peaked circuit sampling](https://arxiv.org/abs/2404.14493)" (arXiv:2404.14493), are a prime starting point.

**What makes them ideal?** Despite their apparent complexity, Peaked Circuits possess a crucial property: verifiability. Similar to how Google's quantum supremacy circuits demonstrated a computational task that was hard for classical computers but verifiable by specific quantum properties, Peaked Circuits allow us to validate a miner's output as it compares to an original known problem state. This verifiability is key to establishing trust and accuracy in a decentralized quantum network from day one.

### The Challenges To Overcome
Quantum computing is powerful, but it's not without its significant hurdles. We're staring these challenges down, head-on:

#### Quantum Processing Units (QPUs)
Real quantum hardware, while awe-inspiring, comes with practical limitations that can hinder adoption:

- **High Error Rates:** Current QPUs are noisy. Quantum states are fragile, leading to errors that propagate through computations.

- **Pervasive Noise:** Environmental and hardware-induced noise consistently interferes with quantum operations, making stable computation a significant challenge.

- **High Operational Costs:** Accessing and running computations on real QPUs is prohibitively expensive for many researchers and developers.

- **Long Queues:** Demand often outstrips supply, leading to long waiting times for access to premium quantum hardware.

See our [Hardware run](qbittensor/miner/docs/HARDWARE_CASESTUDY.md) documenting a 12-qubit peaked circuit execution on IBM's real quantum hardware.

#### Quantum Simulators

While simulators offer a more accessible alternative, they too present their own set of complexities:

- **Demanding Computational Resources:** Full state-vector simulations, which track every possible quantum state, quickly become computationally intractable. Beyond approximately 50-60 qubits, the memory and processing power required become astronomical.

- **Accuracy Limitations:** Many simulators rely on approximations or specialized algorithms (e.g., restricted quantum operations) to handle larger circuits, which can introduce inaccuracies.

- **Varying Scalability:** The ability of a simulator to handle deeper circuits or more qubits varies significantly based on its underlying algorithmic approach.

- **Diverse Algorithmic Approaches:**

    - **State-Vector Simulators:** High accuracy but poor scalability.

    - **Tensor Network Simulators:** Better scalability for certain circuit structures but introduce approximations and can still be resource-intensive.

    - **Clifford Simulators:** Excellent scalability for specific types of circuits (Clifford circuits) but limited in generality.

    - **Proprietary Simulators:** Beyond these open approaches, a handful of enterprising startups have also developed their own propietary simulation techniques to try to better optimize the tradeoffs.

Each approach comes with unique trade-offs in terms of speed, memory, and applicability.

Your goal is to navigate all of these complexities to outpuerform other miners and get the the right answer faster through pulling together the right hardware and the right algorithms, and doing any amount of pre-processing and post-processing, to get to the right answer quicker.

### Miner Activities

In this initial phase, our miners are the pioneers on the computational frontier. Their primary objective is to identify and optimize the killer tech stack – the most effective combination of hardware and software for accurate and efficient circuit execution.

Miner tasks might include:

- **Resource Exploration & Optimization:** This involves an ongoing process of efficiently utilizing and rigorously comparing various quantum technologies. A strong emphasis will be placed on optimizing quantum simulators to push their performance boundaries.

- **Accurate Execution:** Despite the inherent limitations of current QPUs and the various quirks of simulators, miners will be tasked with consistently finding the correct "peak bitstring."

- **Performance Enhancement:** Miners are incentivized to discover and implement the most efficient and reliable methods for achieving accurate quantum results. This includes unearthing and applying subtle optimization tricks, clever resource management, and innovative approaches to computation.

To get started as a Miner consult our [Setup Guide](SETUP_GUIDE.md).

Also check out our [Custom Solver Guide](qbittensor/miner/docs/CUSTOM_SOLVER.md) and [Custom Simulator/Processor Guide](qbittensor/miner/docs/CUSTOM_SIMS_PROCESSORS.md).

### Validator Activities

Validators, with high-end hardware, accomplish the following tasks:

- **Generate Peaked Quantum Circuits:** These circuits are generated with a hidden message that miners solve through quantum simulations.

- **Validate Miners Performance:** Miners solve the circuit and return the initial message; Validators verify that the message is correct and grade Miners accordingly.

- **Distribute Certificates:** Validators provide Miners with a Certificate that will be distributed to all other Validators. Not only does this allow Miners to run very difficult circuits with hours- or days-long runtime, but it also improves VTrust for each Validator, as everybody has the same solutions.

**A note on Validator-Miner collusion**
We have heard stories of Validators helping Miners, or mining themselves with priviliged information. In an effort to be 100% transparent, this is a strong attack vector on Quantum and is being constantly monitored. We have internal scripts to monitor this, and a hardcoded whitelist of Validators (right now it is the top Validators by stake). If you are found to be behaving maliciously you will publicly outed and removed from our Whitelist.

To get started as a Validator consult our [Setup Guide](SETUP_GUIDE.md).

## Scoring

The main scoring function is a weighted sum of **entanglement entropy** and a **size function** for a given circuit of **n** qubits. 

- Miners must first correctly identify the peaked state in order to receive a score.

-  The entanglement entropy (weighted at **30%**) rewards miners for solving circuits with more "quantumness", while the size function (weighted at **70%**) rewards larger qubit counts.

-  The size function is designed to reward miners more aggressively for crossing the 32-qubit barrier, below which the score curve is linear.

| Symbol | Formula | Notes |
|--------|---------|-------|
| **G** (size) | piece-wise (see below) | Rewards larger circuits. |
| **Decay** | `Cₜ = C·e^(−λ·t)` | `λ = ln 2 / 24 h` ⇒ half-life 72 h. |

---

#### Size Function

```python
knee       = 32 # inflection point (qubits)
target     = 50 # maximum score at 50 qubits
min_G      = 0.15 # minimum score at 12 qubits
knee_score = 0.40 # score at the knee
exponential_base = 1.7

if n <= knee: # 12 ≤ n ≤ 32
    t = (n − 12) / (knee − 12)
    G = min_G + (knee_score − min_G) * t # linear 0.15 → 0.40
else: # n > 32
    t = (n − knee) / (target − knee)
    G = knee_score + (1 − knee_score) * t**1.7 # smooth rise 0.40 → 1.00
```
The Knee value is set at 32 qubits, creating a piecewise function that disproportionately rewards Miners that can run circuits larger than 32 qubits. Memory requirements make higher qubit counts beyond this size increasingly difficult. **Miners will need clever ideas and/or SOTA hardware to go much beyond 32 qubits for Peaked Circuits.**

Hidden Stabilizer circuits are scored similarly with a knee at 26. 

A final score is calculated by combining the Peaked Circuit score with a weight of 80% and a Hidden Stabilizer score with a weight of 20%:

```python
combined = 0.8 * peaked_norm + 0.2 * hstab_norm
```

## Strategic Roadmap: Our Future Vision
qBitTensor is just getting started. Our strategic roadmap outlines a progression towards a fully decentralized, high-impact quantum platform:

- **More Circuit Types:** Once we've mastered Peaked Circuits, we will expand to include a wider array of verifiable quantum circuits. This will gradually broaden the scope of problems our network can tackle, building on our foundational understanding of verifiable computation.

- **Real-World Problems:** The next goal is to transition from abstract circuit execution to using circuit execution tech stacks in solving practical, high-impact problems. We envision the network contributing to critical advancements in domains such as:

    - **Quantum Cryptography:** Developing and testing quantum cryptographic algorithms, and challenging classical methods.

    - **Quantum Finance:** Optimizing portfolios, pricing derivatives, and simulating financial markets with quantum methods.

    - **Quantum Chemistry:** Accelerating drug discovery, material science, and molecular simulations.

- **Decentralized Quantum Platform:** Our long-term vision is to evolve into a comprehensive decentralized quantum platform. External users will be able to submit their own quantum circuits, and these will be distributed and executed by our network of miners. This will democratize access to distributed quantum computational power, leveraging both advanced simulators and, eventually, real QPUs as they become more robust and accessible.

We're building a future where quantum computing is not just for a select few, but a powerful, collaborative tool accessible to everyone. Join us on this quantum journey!
