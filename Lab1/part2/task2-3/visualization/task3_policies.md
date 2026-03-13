### Q-Learning Learned Policy

| Y\X | 0 | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|---|
| 4 | RIGHT | RIGHT | RIGHT | RIGHT | GOAL |
| 3 | UP | UP | OBS | UP | UP |
| 2 | RIGHT | RIGHT | RIGHT | UP | UP |
| 1 | UP | UP | OBS | UP | UP |
| 0 | UP | RIGHT | RIGHT | UP | UP |

## Similarity with Optimal Policy

- **Reference**: Value Iteration optimal policy (Task 2-1)
- **Evaluated states**: 22 (excludes obstacles and goal)
- **Matches**: 21 / 22
- **Similarity**: **95.5%**
- **Mismatched states**:
  - (0,0) learned UP, optimal RIGHT


## Convergence Speed Analysis

**Convergence criteria** (both must hold for the rolling window):
1. **Policy stability**: every valid state's greedy action is constant or
   switches between at most 2 actions across the entire window.
2. **Q-value stability**: for every (s,a), |Q(end) - Q(start)| <= 1 over the window.

**Valid states tracked**: 22
**Rolling window**: 1500 episodes

**Convergence point**: Episode `5700`

See `QLearning_Convergence_Analysis.png` for plot.

