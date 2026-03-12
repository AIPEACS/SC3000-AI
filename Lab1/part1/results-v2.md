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

## Task 3c: A* — Haversine + energy-aware, all-neighbours online

Collects `(h[nb], edge_cost)` for **all** valid neighbours at each expansion.

- Final regression: cost ≈ 24.150158 × haversine_dist + 11.34
- Pearson correlation: 0.979317

- Shortest distance: 150335.55442 m
- Total energy cost: 259087
- Number of nodes in path: 122
- Number of states visited: 38882

## Task 3d: A* — Pythagorean + energy-aware, all-neighbours online

Collects `(h[nb], edge_cost)` for **all** valid neighbours at each expansion.

- Final regression: cost ≈ 20.382032 × pythagorean_dist + 114.90
- Pearson correlation: 0.952605

- Shortest distance: 150784.60722 m
- Total energy cost: 287931
- Number of nodes in path: 123
- Number of states visited: 42487

## Task 3e: A* — Haversine heuristic + energy-aware, real-dist regression

Collects `(real_road_dist, edge_cost)` for **all** valid neighbours; regression fits `cost ≈ a·road_dist + b`.

- Final regression: cost ≈ 2.242405 × haversine_dist + 15.43
- Pearson correlation: 0.961558

- Shortest distance: 150335.55442 m
- Total energy cost: 259087
- Number of nodes in path: 122
- Number of states visited: 28021

## Task 3f: A* — Pythagorean heuristic + energy-aware, real-dist regression

Collects `(real_road_dist, edge_cost)` for **all** valid neighbours; regression fits `cost ≈ a·road_dist + b`.

- Final regression: cost ≈ 2.240185 × pythagorean_dist + 17.67
- Pearson correlation: 0.961076

- Shortest distance: 150335.55442 m
- Total energy cost: 259087
- Number of nodes in path: 122
- Number of states visited: 27749

## Comparison: states visited and path accuracy

| Algorithm                                      | States visited | Reduction vs UCS | Path optimality    |
|------------------------------------------------|----------------|------------------|--------------------|
| Task 2  UCS constrained (optimal)              |          30267 | --               | 100.00% (baseline) |
| Task 3a A* Haversine                           |          29111 |     3.8%          | 100.00%             |
| Task 3b A* Pythagorean                         |          29000 |     4.2%          | 100.00%             |
| Task 3c A* Haversine + EA all-nb online        |          38882 |   -28.5%          | 100.00%             |
| Task 3d A* Pythagorean + EA all-nb online      |          42487 |   -40.4%          | 99.70%             |
| Task 3e A* Haversine heuristic + EA real-dist   |          28021 |     7.4%          | 100.00%             |
| Task 3f A* Pythagorean heuristic + EA real-dist |          27749 |     8.3%          | 100.00%             |
