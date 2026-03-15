### Q-Learning Learned Policy

| X\Y | 0 | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|---|
| 4 | RIGHT | RIGHT | RIGHT | RIGHT | GOAL |
| 3 | RIGHT | RIGHT | RIGHT | RIGHT | UP |
| 2 | UP | OBS | UP | OBS | UP |
| 1 | UP | RIGHT | UP | RIGHT | UP |
| 0 | RIGHT | RIGHT | RIGHT | RIGHT | UP |

## Similarity with Optimal Policy

- **Reference**: Value Iteration optimal policy (Task 2-1)
- **Evaluated states**: 22 (excludes obstacles and goal)
- **Matches**: 20 / 22
- **Similarity**: **90.9%**
- **Mismatched states**:
  - (0,0) learned RIGHT, optimal UP
  - (0,2) learned RIGHT, optimal UP


## Convergence Speed Analysis

**Convergence criteria** (both must hold for the rolling window):
1. **Policy stability**: every valid state's greedy action is constant or
   switches between at most 2 actions across the entire window.
2. **Q-value stability**: for every (s,a), |Q(end) - Q(start)| <= 1 over the window.

**Valid states tracked**: 22
**Rolling window**: 1500 episodes

**Convergence point**: Episode `5300`

See `QLearning_Convergence_Analysis.png` for plot.

