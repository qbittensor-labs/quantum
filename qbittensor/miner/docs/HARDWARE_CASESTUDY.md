# Peaked Circuits on Real Quantum Hardware

This file documents the execution of a 12-qubit peaked circuit on IBM's `ibm_brisbane` quantum processor. We used the same circuit generation code that forms the peaked circuits miners are given, but used here on a smaller scale. Real hardware noise limits the number of qubits and circuit depth we can run. For this case we generated a 12-qubit peaked circuit with a target state probability of **34.6575%** and target string **`111011001100`**.

### Job Metadata
| Parameter | Value |
|-----------|--------|
| **Backend** | ibm_brisbane |
| **Hub** | ibm-q |
| **Group** | open |
| **Project** | main |
| **Program** | sampler |
| **Status** | Completed |

*Above is some general details about the backend and primitive used.*

## Circuit Visualization
The circuits given to miners contain **CNOT**, **RZZ**, and other gates that must be transpiled to run on real hardware. The full circuit was transpiled to the following hardware basis gates: **ECR**, **ID**, **RZ**, **SX**, **X** and forced to run on qubits **0-11**. 

Here we visualize the full transpiled circuit. *As you can see it is quite large!*

### Circuit Structure

![Circuit Diagram](circuit_plot.png)

### Physical Layout
| Property | Value |
|----------|-------|
| **Qubits** | 12 |
| **Classical Bits** | 12 |
| **Circuit Depth** | 141 |
| **Total Gates** | 1,322 |
| **Circuit Width** | 24 |

### Gate Composition
| Gate Type | Count | Percentage |
|-----------|--------|------------|
| **RZ** | 636 | 48.1% |
| **SX** | 524 | 39.6% |
| **ECR** | 150 | 11.4% |
| **Measure** | 12 | 0.9% |
| **Barrier** | 1 | 0.1% |

The circuit has 12 qubits, a depth of 141, and a total gate count of 1,322.
In order to get the best results considering measurement error will reduce the peaking probability significantly, the following sampler configuration was used. 

## Sampler Configuration

### Dynamical Decoupling
- **Status**: Enabled
- **Sequence**: XY4
- **Scheduling**: ASAP
- **Slack Distribution**: Middle

### Twirling
- **Gate Twirling**: Enabled
- **Measurement Twirling**: Enabled  
- **Randomizations**: 32

### Execution Settings
- **Initialize Qubits**: True
- **Repetition Delay**: 0.0002 seconds

### Timing Analysis
| Metric | Value | Notes |
|--------|-------|-------|
| **Job Started** | 2025-06-26 18:50:54 UTC | |
| **Job Ended** | 2025-06-27 04:09:06 UTC | |
| **Total Duration** | 9:18:12.271000 | Queue + Execution Time |
| **Estimated Runtime** | 2.47 seconds | Actual quantum execution |
| **Queue Time** | ~9 hours 15 minutes | Time spent waiting in queue |

While the actual execution time was fast, the wait time was over *9 hours!* Below we visualize the raw counts from the job execution and measurement results. 

### Results Distribution
![Results Plot](results_plot.png)

## Measurement Results

### Sampling Statistics
| Metric | Value |
|--------|-------|
| **Total Shots** | 4,096 |
| **Unique Outcomes** | 1,924 |
| **Bits Measured** | 12 |
| **Data Type** | BitArray |

### Distribution Analysis
| Statistic | Value | Context |
|-----------|-------|---------|
| **Most Frequent Outcome** | 295 shots | 7.202% of total |
| **Least Frequent Outcome** | 1 shot | 0.024% of total |
| **Average per Outcome** | 2.1 shots | |
| **Expected (Uniform)** | 1.0 shots | 4096/4096 = 1.0 |
| **Peak Enhancement** | 295.0x | Above uniform expectation |

### Top 10 Most Frequent Outcomes
| Rank | Bitstring | Frequency | Percentage | Notes |
|------|-----------|-----------|------------|-------|
| 1 | **`111011001100`** | 295 | 7.202% | **Target** |
| 2 | `111011001101` | 19 | 0.464% | |
| 3 | `111011000011` | 19 | 0.464% | |
| 4 | `111111001100` | 19 | 0.464% | |
| 5 | `111001111111` | 18 | 0.439% | |
| 6 | `001111001100` | 18 | 0.439% | |
| 7 | `111011000000` | 18 | 0.439% | |
| 8 | `111100111100` | 17 | 0.415% | |
| 9 | `011000111100` | 15 | 0.366% | |
| 10 | `111011000100` | 15 | 0.366% | |

> While we successfully found the peaked state running on real hardware, extremely long queue times and high error rates make any practical use within Bittensor challenging. In the current state real hardware noise and readout error cut our target probability from **34.6%** in simulation to just **7.2%** on device. This error will compound when larger circuits are created; a 12 qubit circuit is much smaller than anything used in the subnet. Furthermore, current coherence times impose a hard ceiling on how deep a peaked circuit can be before decoherence and budget constraints make it infeasible to run. 

> Miners can still use IBM's runtime service using a custom solver and backend if they so choose. An extremely innovative and novel approach for transposing circuits to hardware could set a miner well above the rest with significantly faster compute times compared to classical simulation!