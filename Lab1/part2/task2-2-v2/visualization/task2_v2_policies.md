# Task 2-v2: Monte Carlo Learning (Sliding Window)

## Configuration
- Window size: 1000 most recent returns per (s,a)
- Epsilon-greedy exploration: ¦Å = 0.1
- Stochastic transitions: 0.8 intended, 0.1 each perpendicular.

## Learned Policy

## Monte Carlo v2 ¡ª Learned Policy

| Y\X | 0 | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|----|
| 4 | RIGHT | RIGHT | RIGHT | RIGHT | GOAL |
| 3 | UP | UP | OBS | UP | UP |
| 2 | UP | RIGHT | RIGHT | UP | UP |
| 1 | RIGHT | UP | OBS | UP | UP |
| 0 | RIGHT | RIGHT | RIGHT | UP | UP |

## Legend
- `UP` / `DOWN` / `LEFT` / `RIGHT` = action
- `OBS` = Obstacle
- `GOAL` = Goal state (4,4)

## Similarity with Optimal Policy

- **Reference**: Value Iteration optimal policy (Task 2-1)
- **Evaluated states**: 22 (excludes obstacles and goal)
- **Matches**: 20 / 22
- **Similarity**: **90.9%**
- **Mismatched states**:
  - (0,1) learned RIGHT, optimal UP
  - (0,2) learned UP, optimal RIGHT

