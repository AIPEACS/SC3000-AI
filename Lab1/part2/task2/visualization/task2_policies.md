# Task 2: Monte Carlo Learning

## Problem
5x5 Grid World with stochastic transitions. Unknown environment model.
Stochastic transitions: 0.8 intended, 0.1 each perpendicular direction.
Epsilon-greedy exploration with epsilon=0.1.

## Learned Policy

## Monte Carlo - Learned Policy

| Y\X | 0 | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|----|
| 4 | RIGHT | UP | RIGHT | RIGHT | GOAL |
| 3 | DOWN | RIGHT | OBS | RIGHT | UP |
| 2 | DOWN | RIGHT | RIGHT | RIGHT | UP |
| 1 | RIGHT | UP | OBS | RIGHT | UP |
| 0 | DOWN | RIGHT | RIGHT | RIGHT | UP |

## Legend
- `UP` = Move up
- `DOWN` = Move down
- `LEFT` = Move left
- `RIGHT` = Move right
- `OBS` = Obstacle
- `GOAL` = Goal state (4,4)
