# Lab 1 Description

# part 1

## Graph Search Algorithms on Road Networks

Part 1 explores uninformed and informed graph search algorithms on a **real-world road network** with the goal of finding optimal paths given multiple constraints.

### Problem Setup
- **Graph**: Real-world locations (nodes) connected by roads (weighted edges)
- **Edge weights**: 
  - `Dist.json` — distance (in decimetres) for each edge
  - `Cost.json` — energy cost for each edge
- **Search parameters**:
  - Start node: "1"
  - Goal node: "50"
  - Energy budget: 287,932 units
- **Coordinates**: `Coord.json` stores GPS coordinates (lon, lat) as integers = degrees × 10⁶

### Task 1: UCS (Relaxed — No Energy Constraint)

**Objective**: Find the shortest path from start to goal **ignoring energy cost**.

#### Algorithm: Uniform Cost Search (UCS)
- **Priority queue**: Ordered by accumulated distance (cost)
- **Expansion**: Always expands the node with minimum cumulative cost first
- **Optimality**: Guaranteed to find shortest path under non-negative costs
- **State**: Single node (no energy tracking needed)

**Results**:
- Shortest distance: **148,648.6**
- Path length: **122 nodes**
- States visited: **5,304**

---

### Task 2: UCS with Energy Constraint

**Objective**: Find the shortest path while staying within the energy budget of 287,932 units.

#### Algorithm: Constrained UCS with Pareto Dominance Pruning
- **Expanded state**: `(node, energy_accumulated)`
- **Priority queue**: Ordered by accumulated distance
- **Constraint**: Prune edges that would exceed the energy budget
- **Optimality technique — Pareto dominance**:
  - At each node, maintain a **Pareto front** of non-dominated (distance, energy) labels
  - A new label `(d', e')` is discarded if any existing label `(d, e)` satisfies: $d ≤ d'$ AND $e ≤ e'$
  - Labels dominated by the new one are removed
  - This prunes suboptimal paths early, reducing the search space

**Results**:
- Shortest distance: **150,335.6** — slightly longer due to energy constraint
- Total energy used: **259,087 units** (within budget of 287,932)
- Path length: **122 nodes**
- States visited: **30,267** — 5.7× more than Task 1 due to expanded state space

---

### Task 3a: A* with Haversine Heuristic

**Objective**: Find the shortest energy-constrained path using **informed search** with Haversine (great-circle) heuristic.

#### Haversine Heuristic: Why Admissible?

A* requires an **admissible heuristic** $h(n)$ that never overestimates the true remaining cost.

Since the graph represents real-world locations with GPS coordinates, the **Haversine formula** computes the great-circle (straight-line) distance between two points on Earth's surface:

$$a = \sin^2\!\left(\frac{φ_2 - φ_1}{2}\right) + \cos φ_1 \cdot \cos φ_2 \cdot \sin^2\!\left(\frac{λ_2 - λ_1}{2}\right)$$

$$h(n) = 2R \cdot \arcsin(\sqrt{a})$$

where $R = 6,371,000$ m (Earth's mean radius).

**Admissibility**: Road distances are **always ≥ straight-line distances**, so Haversine never overestimates → admissible.

#### Implementation
```python
R = 63710000.0  # decimetres
lon2, lat2 = (v / 1e6 for v in Coord[goal])
lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

h = {}
for node, vals in Coord.items():
    lon1, lat1 = vals[0] / 1e6, vals[1] / 1e6
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
    h[node] = 2 * R * math.asin(math.sqrt(a))
```

The heuristic is **precomputed once** for all nodes → goal, giving $O(1)$ lookup during A*.

#### Algorithm: A* with Pareto Dominance on Expanded State
- **Priority queue**: Ordered by $f(n) = g(n) + h(n)$ where $g(n)$ = accumulated distance
- **Expanded state**: `(node, energy_accumulated)`
- **Pruning**: Same Pareto dominance as Task 2
- **Optimality**: Guaranteed (Haversine is admissible and A* expands in best-first order)

**Results**:
- Shortest distance: **150,335.6 decimetres** (same as Task 2 — optimal)
- Total energy used: **259,087 units**
- Path length: **122 nodes**
- States visited: **9,552** — **68.4% reduction** vs UCS constrained

---

### Task 3b: A* with Pythagorean/Euclidean Heuristic

**Objective**: Find the shortest energy-constrained path using **Euclidean distance heuristic** (comparison).

#### Euclidean Heuristic
Simple straight-line distance in (lon, lat) space:

$$h(n) = \sqrt{(lon_2 - lon_1)^2 + (lat_2 - lat_1)^2} \times 10^6$$

where coordinates are scaled by $10^6$ to match distance units.

**Note**: This heuristic is **not truly admissible** in geographic coordinates (Euclidean distance is not the true minimum), but it serves as a benchmark.

#### Algorithm
Same A* with expanded state and Pareto dominance as Task 3a, but uses Euclidean distance instead of Haversine.

**Results**:
- Shortest distance: **150,335.6 decimetres** (same optimal path)
- Total energy used: **259,087 units**
- Path length: **122 nodes**
- States visited: **3,271** — **89.2% reduction** vs UCS constrained

---

### Comparison: Efficiency of Search Algorithms

| Algorithm | States Visited | Reduction vs UCS | Path Optimality | Heuristic Quality |
|-----------|---|---|---|---|
| **Task 1: UCS (relaxed)** | 5,304 | -- | 100% (baseline) | None |
| **Task 2: UCS (constrained)** | 30,267 | -- | 100% (optimal under budget) | None |
| **Task 3a: A* Haversine** | 9,552 | **68.4%** ↓ | 100% (optimal) | Haversine (admissible) |
| **Task 3b: A* Euclidean** | 3,271 | **89.2%** ↓ | 100% (optimal) | Euclidean (tighter estimate) |

### Key Insights

1. **Informed search beats uninformed**: A* with Haversine reduces states expanded by 68% by guiding the search toward the goal.
2. **Heuristic quality matters**: Even though Euclidean is not strictly admissible, it's a tighter estimate of road distance than Haversine and achieves 89% reduction.
3. **Pareto dominance pruning**: Critical for multi-objective optimization (minimizing both distance and energy usage).
4. **Energy constraint adds complexity**: Expanding state space from 50 nodes to 30K+ (node, energy) tuples, but informed search still maintains 3-5K expansions.

# part 2

## Reinforcement Learning in Stochastic Grid World

Part 2 explores different reinforcement learning and planning algorithms in a **stochastic 5×5 grid world**.

### Environment

- **Grid**: 5×5, coordinates $(x, y)$ where **x is the vertical axis** (0 = bottom, 4 = top) and **y is the horizontal axis** (0 = left, 4 = right)
- **Start**: $(0, 0)$ — bottom-left
- **Goal**: $(4, 4)$ — top-right
- **Obstacles**: $(2, 1)$ and $(2, 3)$
- **Step cost**: $-1$; **Goal reward**: $+10$ (net $= +9$)
- **Discount factor**: $\gamma = 0.9$
- **Stochastic transitions**: 0.8 probability of intended action, 0.1 each perpendicular direction

All tasks use the same environment defined in `scene_map.py`.

---

### Task 2-1: Planning with Known Model (Value & Policy Iteration)

**Scenario**: The agent **knows** the exact transition probabilities.

#### Value Iteration
- **Update rule**: $V(s) \leftarrow \max_a \sum_{s'} P(s'|s,a)\bigl[R(s') + \gamma V(s')\bigr]$
- **Convergence threshold**: $\delta < 10^{-6}$

#### Policy Iteration
- **Policy Evaluation**: Solve $V^\pi(s) = \sum_a \pi(a|s)\sum_{s'} P(s'|s,a)\bigl[R(s') + \gamma V^\pi(s')\bigr]$
- **Policy Improvement**: $\pi'(s) = \arg\max_a Q^\pi(s,a)$
- **Convergence**: When policy stops changing across a full sweep

Both algorithms produce the **identical** optimal policy:

| X\Y | 0 | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|---|
| **4** | RIGHT | RIGHT | RIGHT | RIGHT | GOAL |
| **3** | RIGHT | RIGHT | RIGHT | RIGHT | UP |
| **2** | UP | OBS | UP | OBS | UP |
| **1** | UP | RIGHT | UP | RIGHT | UP |
| **0** | UP | RIGHT | UP | RIGHT | UP |

**Key insight**: Policy Iteration typically converges in fewer iterations than Value Iteration; both yield the same optimal policy.

---

### Task 2-2: Learning with Unknown Model (Monte Carlo Control)

**Scenario**: Transition probabilities are **unknown**; the agent learns purely from sampled episodes.

#### Algorithm: On-Policy First-Visit MC Control
- **Exploration**: $\epsilon$-greedy with $\epsilon = 0.1$
- **Return**: $G_t = \sum_{k=0}^{T-t} \gamma^k R_{t+k}$ (computed backwards per episode)
- **Q-update**: $Q(s,a) \leftarrow \text{mean of all returns collected for } (s,a)$
- **Data structure**: `returns[(s,a)] = [G_1, G_2, \ldots]` — unbounded list; all historical returns averaged
- **Training**: 20,000 episodes, $\epsilon = 0.1$

#### Results (single run)
- **Policy match vs VI optimal**: **90.9%** (20 / 22 valid states)
- **Mismatched states**: $(0,2)$ learned RIGHT (optimal UP), $(1,2)$ learned RIGHT (optimal UP)

---

### Task 2-2-v2: Monte Carlo with Sliding-Window Returns

**Scenario**: Same as Task 2-2, but Q-values are estimated from only the **most recent 1000 returns** per $(s,a)$.

#### Modification
- **Data structure**: `collections.deque(maxlen=1000)` — oldest return auto-discarded when full
- **Motivation**: Averaging all historical returns weights early (suboptimal) policy samples equally with recent ones. The sliding window discards stale data so Q-values track the current near-optimal policy more faithfully.
- **Training**: 20,000 episodes, $\epsilon = 0.1$, window size = 1000

#### Results (single run)
- **Policy match vs VI optimal**: **90.9%** (20 / 22 valid states)
- **Mismatched states**: $(0,0)$ learned RIGHT (optimal UP), $(1,0)$ learned RIGHT (optimal UP)

---

### Task 2-3: Learning with Unknown Model (Q-Learning)

**Scenario**: Off-policy TD learning — updates Q-values after every single step.

#### Algorithm: Tabular Q-Learning
- **Update rule**: $Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[R(s') + \gamma \max_{a'} Q(s', a') - Q(s,a)\bigr]$
- **Learning rate**: $\alpha = 0.1$; **Exploration**: $\epsilon = 0.1$ (fixed)
- **Off-policy**: bootstraps greedily at next state, decoupled from current exploration policy
- **Training**: 50,000 episodes

#### Convergence Analysis
Convergence is declared when **both** of the following hold over a rolling 1500-episode window:
1. **Policy stability**: every valid state's greedy action switches between at most 2 actions across the window
2. **Q-value stability**: $|Q_{\text{end}}(s,a) - Q_{\text{start}}(s,a)| \leq 1$ for all $(s,a)$

**Convergence point: Episode 5,300** (out of 50,000 trained)

#### Results (single run)
- **Policy match vs VI optimal**: **90.9%** (20 / 22 valid states)
- **Mismatched states**: $(0,0)$ learned RIGHT (optimal UP), $(0,2)$ learned RIGHT (optimal UP)

#### Learned Policy

| X\Y | 0 | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|---|
| **4** | RIGHT | RIGHT | RIGHT | RIGHT | GOAL |
| **3** | RIGHT | RIGHT | RIGHT | RIGHT | UP |
| **2** | UP | OBS | UP | OBS | UP |
| **1** | UP | RIGHT | UP | RIGHT | UP |
| **0** | RIGHT | RIGHT | RIGHT | RIGHT | UP |

---

### MC v2 vs Q-Learning: Statistical Comparison (50 runs × 20,000 episodes)

To fairly compare Monte Carlo v2 and Q-Learning under identical conditions, both algorithms were trained 50 independent times with 20,000 episodes each and evaluated against the VI optimal policy (`compare_MC_QL.py`).

| Metric | Monte Carlo v2 | Q-Learning |
|--------|:--------------:|:----------:|
| **Mean accuracy** | 85.09% | **88.82%** |
| **Standard deviation** | 10.12% | **6.25%** |
| **Variance** | 102.51 | **39.02** |
| Min | 63.64% | 68.18% |
| Max | 100.00% | 100.00% |
| Median | 86.36% | **90.91%** |

**Key findings**:
1. **Q-Learning is more accurate**: +3.7 percentage points higher mean accuracy than MC v2 at equal episode counts.
2. **Q-Learning is more stable**: variance 2.6× lower than MC v2, meaning its policy quality is far more consistent run-to-run.
3. **MC v2 has higher variance**: unbounded averaging of *all* historical returns under any policy introduces noise; the sliding window mitigates this vs plain MC but Q-Learning's per-step TD updates still converge faster.
4. **Q-Learning converges earlier**: declared convergent at episode 5,300 — meaning 14,700 of the 20,000 episodes are consolidation, whereas MC v2 continues accumulating returns throughout.

---

### Summary: Algorithm Comparison

| Algorithm | Model | Data | Update | Match (single) | Mean ± Std (50 runs) |
|-----------|-------|------|--------|:--------------:|:--------------------:|
| **Value Iteration** | Known | DP | All states / iter | 100% (reference) | — |
| **Policy Iteration** | Known | DP | Policy sweeps | 100% (reference) | — |
| **Monte Carlo (2-2)** | Unknown | Full episodes | End of episode | 90.9% | — |
| **Monte Carlo v2 (2-2-v2)** | Unknown | Full episodes (window=1000) | End of episode | 90.9% | 85.1% ± 10.1% |
| **Q-Learning (2-3)** | Unknown | Single steps (TD) | Every step | 90.9% | **88.8% ± 6.3%** |

---
