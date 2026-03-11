### Q-Learning Learned Policy

| Y\X | 0 | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|---|
| 4 | RIGHT | RIGHT | RIGHT | RIGHT | GOAL |
| 3 | RIGHT | UP | OBS | UP | UP |
| 2 | RIGHT | RIGHT | RIGHT | UP | UP |
| 1 | UP | UP | OBS | RIGHT | UP |
| 0 | RIGHT | RIGHT | RIGHT | UP | UP |

## Similarity with Optimal Policy

- **Reference**: Value Iteration optimal policy (Task 2-1)
- **Evaluated states**: 22 (excludes obstacles and goal)
- **Matches**: 20 / 22
- **Similarity**: **90.9%**
- **Mismatched states**:
  - (0,3) learned RIGHT, optimal UP
  - (3,1) learned RIGHT, optimal UP

