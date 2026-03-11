### Q-Learning Learned Policy

| Y\X | 0 | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|---|
| 4 | RIGHT | RIGHT | RIGHT | RIGHT | GOAL |
| 3 | UP | UP | OBS | UP | UP |
| 2 | UP | UP | RIGHT | RIGHT | UP |
| 1 | UP | UP | OBS | UP | UP |
| 0 | UP | RIGHT | RIGHT | UP | UP |

## Similarity with Optimal Policy

- **Reference**: Value Iteration optimal policy (Task 2-1)
- **Evaluated states**: 22 (excludes 2 obstacles and goal)
- **Matches**: 18 / 22
- **Similarity**: **81.8%**
- **Mismatched states**:
  - (0,0) learned UP, optimal RIGHT
  - (0,2) learned UP, optimal RIGHT
  - (1,2) learned UP, optimal RIGHT
  - (3,2) learned RIGHT, optimal UP
