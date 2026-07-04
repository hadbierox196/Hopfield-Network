#  Hopfield Network: Associative Memory & Energy Landscapes

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243.svg)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A from-scratch implementation of a **Hopfield network** — a classic model of associative memory that stores patterns as stable, low-energy states and recalls the closest one even from a noisy or partial cue.

---

## What it does

- Implements a Hopfield network with Hebbian learning and asynchronous/synchronous updates
- Recalls stored patterns from noisy cues and measures recall accuracy
- Visualizes the network's **energy landscape** in 2D/3D using PCA
- Measures **storage capacity** and compares it to the classical 0.14N theoretical limit
- Shows how recall degrades ("interference") as more patterns are packed into the same network
- Implements a **sparse-coding variant** and compares it against the standard dense network

---

## Background

A Hopfield network is a recurrent network where every neuron connects to every other neuron. Patterns are stored by shaping an **energy landscape**: each stored pattern becomes a local minimum, so starting from a noisy or incomplete version of a pattern, the network's dynamics roll "downhill" into the nearest stored memory.

Its two most famous limits, both explored here:

| Property | Meaning |
|---|---|
| **Noise tolerance** | How corrupted a cue can be while still recalling the right pattern |
| **Capacity** | How many patterns can be stored before recall starts failing (~0.14 × N neurons) |

---

## Installation

```bash
pip install numpy matplotlib scikit-learn scipy seaborn tqdm
```

## Usage

```bash
python hopfield_network.py
```

Runs all experiments in sequence and saves figures to the working directory (~2–4 minutes total).

### Core API

```python
hopfield = HopfieldNetwork(n_neurons=100)
hopfield.train(patterns)                 # Hebbian learning

retrieved, converged, iters, energy = hopfield.retrieve(noisy_probe, mode='async')
overlap = hopfield.pattern_overlap(retrieved, original_pattern)
```

---

## Experiments

| # | Question | Output |
|---|---|---|
| 1 | Basic demo — can the network recall patterns from noisy cues? | console output |
| 2 | How does recall accuracy degrade with noise? | `hopfield_noise_robustness.png` |
| 3 | What does the energy landscape look like? | `hopfield_energy_landscape.png` |
| 4 | How many patterns can the network store before it breaks down? | `hopfield_capacity_analysis.png` |
| 5 | How does interference show up visually as load increases? | `hopfield_interference.png` |
| 6 | Does sparse coding increase capacity vs. dense coding? | comparison plot |

---

## Results (typical)

| Metric | Value |
|---|---|
| Theoretical capacity (0.14N, N=100) | ~14 patterns |
| Empirical capacity (80% accuracy threshold) | close to theoretical prediction |
| Noise tolerance | strong recall up to ~20–30% flipped bits, degrading sharply beyond it |
| Sparse vs. dense coding | sparse patterns support higher effective capacity |

**Takeaway:** the network behaves exactly as Hopfield/Amit-Gutfreund-Sompolinsky theory predicts — a sharp capacity cliff, graceful degradation under noise, and improved storage with sparser patterns.

---

## Math, briefly

**Hebbian weight rule:** `W = (1/P) Σₚ ξᵖξᵖᵀ`, diagonal zeroed

**Energy function:** `E(s) = -½ sᵀWs`

**Neuron update:** `sᵢ ← sign(Σⱼ Wᵢⱼ sⱼ)`

**Theoretical capacity:** `P_max ≈ 0.14N`

---

## Roadmap

- Modern Hopfield networks (continuous states, exponential capacity)
- Structured/correlated pattern sets instead of random patterns
- Basin-of-attraction size estimation
- Simulated annealing / stochastic (Boltzmann-style) updates

---

## License

MIT — see [LICENSE](LICENSE).

## References

- Hopfield, J. J. (1982) — *Neural networks and physical systems with emergent collective computational abilities*, PNAS
- Amit, Gutfreund & Sompolinsky (1985) — *Storing infinite numbers of patterns in a spin-glass model of neural networks*, Physical Review Letters
- Hertz, Krogh & Palmer — *Introduction to the Theory of Neural Computation*
