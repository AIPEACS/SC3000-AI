# Heuristic Comparison: Haversine vs Pythagorean/Euclidean

## Task 1: UCS (relaxed — no energy constraint)

- Shortest distance: 148648.63722 m
- Number of nodes in path: 122
- Number of states visited: 5304

## Task 2: UCS (energy-constrained shortest path)

- Shortest distance: 150335.55442 m
- Total energy cost: 259087
- Number of nodes in path: 122
- Number of states visited: 30267

## Task 3a: A* — Haversine heuristic (great-circle distance)

- Shortest distance: 150335.55442 m
- Total energy cost: 259087
- Number of nodes in path: 122
- Number of states visited: 29111

## Task 3b: A* — Pythagorean/Euclidean heuristic

Formula: `sqrt(dlat^2 + dlon^2) * 111111 m/degree`

- Shortest distance: 150335.55442 m
- Total energy cost: 259087
- Number of nodes in path: 122
- Number of states visited: 29000

## Task 3c: A* — Haversine + energy-aware `h(n) * B / (B - (a*h(n) + b))`

- Linearity (Haversine): cost ≈ 22.559548 × haversine_dist + 207.30
- Pearson correlation: 0.963689

- Shortest distance: 150335.55442 m
- Total energy cost: 259087
- Number of nodes in path: 122
- Number of states visited: 20728

## Task 3d: A* — Pythagorean + energy-aware `h(n) * B / (B - (a*h(n) + b))`

- Linearity (Pythagorean): cost ≈ 19.390584 × pythagorean_dist + 223.52
- Pearson correlation: 0.953631

- Shortest distance: 150335.55442 m
- Total energy cost: 259087
- Number of nodes in path: 122
- Number of states visited: 11530

## Comparison: states visited and path accuracy

| Algorithm                             | States visited | States Visited Reduction | Path optimality |
|---------------------------------------|----------------|------------------|----------------------------|
| Task 2 UCS constrained (optimal)      |          30267 | --               | 100.00% (baseline)         |
| Task 3a A* Haversine                  |          29111 |     3.8%          | 100.00%                       |
| Task 3b A* Pythagorean                |          29000 |     4.2%          | 100.00%                       |
| Task 3c A* Haversine + energy-aware   |          20728 |    31.5%          | 100.00%                       |
| Task 3d A* Pythagorean + energy-aware |          11530 |    61.9%          | 100.00%                       |
