# Lab1 descrption

# part 1

## A* Heuristic — Haversine Formula

A* requires an **admissible heuristic** h(n): an estimate of the remaining distance from node n to the goal that **never overestimates** the true cost.

Since the graph represents real-world locations with GPS coordinates, the heuristic used is the **Haversine formula** — the great-circle (straight-line) distance between two points on the Earth's surface.

### Why Haversine?
- Road distances are always ≥ straight-line distances, so Haversine never overestimates → **admissible**.
- It accounts for the Earth's curvature, making it more accurate than simple Euclidean distance for geographic coordinates.

### Formula

Given two points with latitude/longitude $(φ_1, λ_1)$ and $(φ_2, λ_2)$ in radians:

$$a = \sin^2\!\left(\frac{φ_2 - φ_1}{2}\right) + \cos φ_1 \cdot \cos φ_2 \cdot \sin^2\!\left(\frac{λ_2 - λ_1}{2}\right)$$

$$d = 2R \cdot \arcsin(\sqrt{a})$$

where $R = 6{,}371{,}000$ m (Earth's mean radius).

### Implementation detail
Coordinates in `Coord.json` are stored as integers = degrees × 10⁶, so they are divided by `1e6` before conversion to radians:

```python
R = 6371000.0
lon1, lat1 = vals[0] / 1e6, vals[1] / 1e6
dlat = lat2_r - lat1_r
dlon = lon2_r - lon1_r
a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
h[node] = 2 * R * math.asin(math.sqrt(a))
```

The heuristic is precomputed once for all nodes → goal before the search begins, giving O(1) lookup during A*.

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