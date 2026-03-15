# Task 2-v2: Monte Carlo Learning (Sliding Window)

## Configuration
- Window size: 1000 most recent returns per (s,a)
- Epsilon-greedy exploration: e = 0.1
- Stochastic transitions: 0.8 intended, 0.1 each perpendicular.

## Learned Policy

## Monte Carlo v2 — Learned Policy

| X\Y | 0 | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|----|
| 4 | RIGHT | RIGHT | RIGHT | RIGHT | GOAL |
| 3 | RIGHT | RIGHT | RIGHT | RIGHT | UP |
| 2 | UP | OBS | UP | OBS | UP |
| 1 | RIGHT | RIGHT | UP | RIGHT | UP |
| 0 | RIGHT | RIGHT | UP | RIGHT | UP |

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
  - (0,0) learned RIGHT, optimal UP
  - (1,0) learned RIGHT, optimal UP

