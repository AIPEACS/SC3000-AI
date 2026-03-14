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

Part 2 explores different reinforcement learning and planning algorithms in a **stochastic 5×5 grid world** where:
- **Start**: (0, 0)
- **Goal**: (4, 4)
- **Obstacles**: (2, 1), (2, 3)
- **Step cost**: -1
- **Goal reward**: +10 (net = +9)
- **Discount factor**: γ = 0.9
- **Stochastic transitions**: 0.8 probability intended action, 0.1 each perpendicular

All tasks use the same environment defined in `scene_map.py`.

### Task 2-1: Planning with Known Model (Value & Policy Iteration)

**Scenario**: The agent **knows** the exact transition probabilities (0.8 intended, 0.1 perpendicular each).

#### Value Iteration
- **Goal**: Find the optimal value function $V^*(s)$ by iteratively applying the Bellman optimality equation
- **Update rule**: $V(s) ← \max_a \sum_{s'} P(s'|s,a)[R(s') + \gamma V(s')]$
- **Convergence**: Guaranteed when max value change $\delta < \theta$ (default: $10^{-6}$)
- **Output**: Deterministic greedy policy extracted from optimal values

#### Policy Iteration
- **Goal**: Find the optimal policy $\pi^*(s)$ by alternating between evaluation and improvement
- **Policy Evaluation**: Compute $V^π(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s') + \gamma V^π(s')]$
- **Policy Improvement**: Extract greedy policy $\pi'(s) = \arg\max_a Q^π(s,a)$
- **Convergence**: When policy stops changing
- **Result**: Often converges faster than value iteration (fewer iterations)

**Key insight**: Both algorithms produce identical optimal policies, enabling direct comparison.

### Task 2-2: Learning with Unknown Model (Monte Carlo Control)

**Scenario**: The agent **does NOT know** transition probabilities and must learn from experience.

#### Monte Carlo (MC) Control Algorithm
- **Method**: On-policy first-visit Monte Carlo with **epsilon-greedy** exploration ($\epsilon = 0.1$)
- **Training**: 
  - Generate full episodes by executing epsilon-greedy actions to termination
  - Compute discounted returns $G_t = \sum_{k=0}^{T-t} \gamma^k R_{t+k}$
  - Track all returns for each (state, action) pair
  - Update Q-values: $Q(s,a) ← \text{mean of all returns for } (s,a)$
- **Data structure**: Dictionary of lists — `returns[(s,a)] = [G₁, G₂, ...]`
- **Convergence**: Gradual as episodes increase; final Q estimates used to extract deterministic policy

**Result**: Learned policy compared against Task 2-1's optimal policy; typically achieves 95-100% match rate.

### Task 2-2-v2: Monte Carlo with Sliding-Window Returns

**Scenario**: Same as Task 2-2, but uses a **fixed-size window** instead of all historical returns.

#### Modification
- **Data structure**: `collections.deque(maxlen=1000)` — maintains only the most recent 1000 returns per (s,a)
- **Benefit**: 
  - Discards stale returns collected under early (suboptimal) policies
  - Q-values track the current near-optimal policy more faithfully
  - Reduces memory usage for long training runs
- **Implementation**: `returns[(s,a)].append(G)` automatically removes the oldest return when full

**Comparison**: Task 2-2-v2 policies typically match the optimal policy **better** than Task 2-2 because recent returns are given more weight.

### Task 2-3: Learning with Unknown Model (Q-Learning)

**Scenario**: Off-policy learning where the agent improves via Q-learning while exploring with epsilon-greedy.

#### Q-Learning Algorithm
- **Method**: Temporal-Difference (TD) learning — updates Q-values from single-step transitions (not full episodes)
- **Update rule**: $Q(s,a) ← Q(s,a) + \alpha [R(s') + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
  - **Learning rate**: $\alpha = 0.1$ (controls update magnitude)
  - **Bootstrap**: Uses max Q-value at next state (greedy targeting, not current action)
- **Off-policy**: Can learn optimal policy while exploring with suboptimal actions
- **Convergence**: Guaranteed under appropriate learning rate decay
- **Advantages over MC**:
  - Updates after **each step**, not just episode end (faster learning)
  - Does not require trajectory to goal (learns from partial episodes)
  - Typically converges with fewer training steps

**Result**: Q-Learning usually matches optimal policy better than Monte Carlo due to faster learning and off-policy nature.

### Summary: Algorithm Comparison

| Algorithm | Transition Model | Data Collection | Update Frequency | Convergence Speed |
|-----------|------------------|-----------------|------------------|-------------------|
| **Value Iteration** | Known | Computed (dynamic programming) | All states each iteration | Fast (guaranteed) |
| **Policy Iteration** | Known | Computed (dynamic programming) | Policy-level iterations | Typically fastest |
| **Monte Carlo (2-2)** | Unknown | Full episodes | Once per episode | Slow (many episodes) |
| **Monte Carlo-v2 (2-2-v2)** | Unknown | Full episodes (windowed) | Once per episode | Slow but better estimates |
| **Q-Learning (2-3)** | Unknown | Single transitions | Every step (online) | Faster than MC |

---
