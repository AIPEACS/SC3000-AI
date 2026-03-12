# Heuristic Comparison: Haversine vs Pythagorean/Euclidean

## Task 1: UCS (relaxed — no energy constraint)

- Shortest distance: 148648.63722
- Number of nodes in path: 122
- Number of states visited: 5304

## Task 2: UCS (energy-constrained shortest path)

- Shortest distance: 150335.55442
- Total energy cost: 259087
- Number of nodes in path: 122
- Number of states visited: 30267

## Task 3a: A* — Haversine heuristic (great-circle distance)

- Shortest distance: 150335.55442
- Total energy cost: 259087
- Number of nodes in path: 122
- Number of states visited: 9552

## Task 3b: A* — Pythagorean/Euclidean heuristic

- Shortest distance: 150335.55442
- Total energy cost: 259087
- Number of nodes in path: 122
- Number of states visited: 3271

## Comparison: states visited and path accuracy

| Algorithm                                      | States visited | Reduction vs UCS | Path optimality    |
|------------------------------------------------|----------------|------------------|--------------------|
| Task 2  UCS constrained (optimal)              |          30267 | --               | 100.00% (baseline) |
| Task 3a A* Haversine                           |           9552 |    68.4%          | 100.00%             |
| Task 3b A* Pythagorean                         |           3271 |    89.2%          | 100.00%             |
