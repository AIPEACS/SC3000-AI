# Task 1: Planning with Known Environment

## Problem
5x5 Grid World with obstacles at (2,1) and (2,3). Discount factor gamma=0.9, step cost=-1, goal reward=+10.

## Optimal Policies

## Value Iteration - Optimal Policy

| Y\X | 0 | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|----|
| 4 | RIGHT | RIGHT | RIGHT | RIGHT | GOAL |
| 3 | UP | UP | OBS | UP | UP |
| 2 | UP | UP | RIGHT | UP | UP |
| 1 | UP | UP | OBS | UP | UP |
| 0 | UP | UP | RIGHT | UP | UP |

## Policy Iteration - Optimal Policy

| Y\X | 0 | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|----|
| 4 | RIGHT | RIGHT | RIGHT | RIGHT | GOAL |
| 3 | UP | UP | OBS | UP | UP |
| 2 | UP | UP | RIGHT | UP | UP |
| 1 | UP | UP | OBS | UP | UP |
| 0 | UP | UP | RIGHT | UP | UP |

## Legend
- `UP` = Move up
- `DOWN` = Move down
- `LEFT` = Move left
- `RIGHT` = Move right
- `OBS` = Obstacle
- `GOAL` = Goal state (4,4)
