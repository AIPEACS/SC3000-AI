import heapq
import json
import math
from pathlib import Path

START = "1"
GOAL = "50"
ENERGY_BUDGET = 287932
DEGREE_CHANGE_COEFF = 1.11111  # coefficient applied per 1 degree of coordinate change


def load_data(base_dir: Path):
    with (base_dir / "G.json").open("r", encoding="utf-8") as f:
        graph = json.load(f)
    with (base_dir / "Coord.json").open("r", encoding="utf-8") as f:
        coord = json.load(f)
    with (base_dir / "Dist.json").open("r", encoding="utf-8") as f:
        dist = json.load(f)
    with (base_dir / "Cost.json").open("r", encoding="utf-8") as f:
        cost = json.load(f)
    return graph, coord, dist, cost


def pythagorean_heuristic_with_degree_coeff(coord, goal, degree_coeff=DEGREE_CHANGE_COEFF):
    """
    Pythagorean (Euclidean) heuristic in coordinate-degree space.

    Coordinate values are stored as integers of degrees * 1e6.
    We convert to degrees first, then scale by `degree_coeff` per degree.
    """
    goal_lon, goal_lat = (v / 1e6 for v in coord[goal])

    heuristic = {}
    for node, vals in coord.items():
        lon, lat = vals[0] / 1e6, vals[1] / 1e6
        dlat = goal_lat - lat
        dlon = goal_lon - lon
        heuristic[node] = math.sqrt(dlat * dlat + dlon * dlon) * 1000000 * degree_coeff
    return heuristic


def astar_constrained_pythagorean(graph, dist, cost, start, goal, budget, heuristic):
    """
    A* on expanded state (node, energy_used) with Pythagorean heuristic.

    Priority  : f = g + h(node)
    Constraint: total energy must not exceed budget
    Pruning   : Pareto dominance on labels (distance, energy) at each node
    """
    pq = [(heuristic.get(start, float("inf")), 0.0, 0, start)]
    best = {start: [(0.0, 0)]}
    parent = {(start, 0): None}
    closed = set()

    while pq:
        _, d, e, node = heapq.heappop(pq)

        if (node, e) in closed:
            continue
        closed.add((node, e))

        if node == goal:
            path, state = [], (goal, e)
            while state is not None:
                path.append(state[0])
                state = parent[state]
            return d, e, path[::-1], len(closed)

        for nb in graph.get(node, []):
            edge = f"{node},{nb}"
            if edge not in dist or edge not in cost:
                continue

            nd = d + float(dist[edge])
            ne = e + int(cost[edge])

            if ne > budget:
                continue

            nb_labels = best.get(nb, [])
            if any(ld <= nd and le <= ne for ld, le in nb_labels):
                continue

            best[nb] = [(ld, le) for ld, le in nb_labels if not (nd <= ld and ne <= le)]
            best[nb].append((nd, ne))

            parent[(nb, ne)] = (node, e)
            nf = nd + heuristic.get(nb, float("inf"))
            heapq.heappush(pq, (nf, nd, ne, nb))

    return float("inf"), -1, [], len(closed)


def main():
    base_dir = Path(__file__).resolve().parent
    graph, coord, dist, cost = load_data(base_dir)

    heuristic = pythagorean_heuristic_with_degree_coeff(coord, GOAL, DEGREE_CHANGE_COEFF)
    shortest_dist, total_energy, path, visited_states = astar_constrained_pythagorean(
        graph, dist, cost, START, GOAL, ENERGY_BUDGET, heuristic
    )

    print("=" * 60)
    print("Task 3 Heursitic Test: A* with Pythagorean + degree coefficient")
    print(f"Degree-change coefficient: {DEGREE_CHANGE_COEFF} per 1 degree")

    if path:
        print(f"Shortest path: {'->'.join(path)}.")
        print(f"Shortest distance: {shortest_dist}.")
        print(f"Total energy cost: {total_energy}.")
        print(f"Number of nodes in path: {len(path)}.")
        print(f"Number of states visited: {visited_states}.")
    else:
        print("No feasible path found within the energy budget.")


if __name__ == "__main__":
    main()
