"""
Random-graph test for Part 1 heuristics.

Generates a random 20-node, ~100-edge directed graph with coordinates
in the same format as the original dataset, then runs:
  - Task 1 : UCS (unconstrained)
  - Task 2 : UCS (energy-constrained, Pareto labelling)
  - Task 3a: A* with Haversine heuristic
  - Task 3b: A* with Pythagorean/Euclidean heuristic

Saves the generated graph to JSON files (G, Coord, Dist, Cost) and
prints a comparison table of states visited / path optimality.
"""

import heapq
import json
import math
import random
from pathlib import Path

# ── Parameters ─────────────────────────────────────────────────────────────────
SEED         = random.randint(0, 999999)
N_NODES      = 150
TARGET_EDGES = 15000      # directed edges
START        = "1"
GOAL         = "120"
R_DM         = 63_710_000.0   # Earth radius in decimetres

# ── Haversine helper ───────────────────────────────────────────────────────────
def _haversine_dm(c1, c2):
    """Straight-line distance between two raw-coord pairs, in decimetres."""
    lon1, lat1 = c1[0] / 1e6, c1[1] / 1e6
    lon2, lat2 = c2[0] / 1e6, c2[1] / 1e6
    φ1, λ1 = math.radians(lat1), math.radians(lon1)
    φ2, λ2 = math.radians(lat2), math.radians(lon2)
    a = math.sin((φ2 - φ1) / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin((λ2 - λ1) / 2) ** 2
    return 2 * R_DM * math.asin(math.sqrt(a))


# ── Random graph generator ─────────────────────────────────────────────────────
def generate_graph(seed=SEED):
    random.seed(seed)
    nodes = [str(i) for i in range(1, N_NODES + 1)]

    # Random coordinates around New York area (same format as original: degrees × 1e6)
    Coord = {
        n: [
            int((-73.50 + random.uniform(-0.05, 0.05)) * 1_000_000),
            int(( 40.70 + random.uniform(-0.05, 0.05)) * 1_000_000),
        ]
        for n in nodes
    }

    # Build edges: random spanning tree first (guarantees connectivity), then fill to target
    edge_set = set()
    perm = nodes[:]
    random.shuffle(perm)
    for i in range(len(perm) - 1):
        edge_set.add((perm[i], perm[i + 1]))
        edge_set.add((perm[i + 1], perm[i]))   # bidirectional spanning tree

    attempts = 0
    while len(edge_set) < TARGET_EDGES and attempts < 50_000:
        u, v = random.sample(nodes, 2)
        edge_set.add((u, v))
        attempts += 1

    # Adjacency list
    G = {n: [] for n in nodes}
    for u, v in edge_set:
        G[u].append(v)

    # Dist = Haversine × road-factor (roads aren't straight lines)
    # Cost = random energy proportional to distance but with variation
    Dist = {}
    Cost = {}
    for u, v in edge_set:
        key = f"{u},{v}"
        sl = _haversine_dm(Coord[u], Coord[v])
        Dist[key] = round(sl * random.uniform(1.05, 1.50), 4)
        Cost[key] = int(Dist[key] * random.uniform(0.5, 2.5))

    return G, Coord, Dist, Cost, nodes


# ── Task 1: UCS (unconstrained) ───────────────────────────────────────────────
def ucs(G, Dist, start, goal):
    pq = [(0.0, start)]
    best = {start: 0.0}
    parent = {start: None}
    visited = 0

    while pq:
        d, node = heapq.heappop(pq)
        if d > best[node]:
            continue
        visited += 1
        if node == goal:
            break
        for nb in G.get(node, []):
            edge = f"{node},{nb}"
            if edge not in Dist:
                continue
            nd = d + Dist[edge]
            if nd < best.get(nb, float("inf")):
                best[nb] = nd
                parent[nb] = node
                heapq.heappush(pq, (nd, nb))

    if goal not in best:
        return float("inf"), [], visited
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    return best[goal], path[::-1], visited


# ── Task 2: UCS (energy-constrained, Pareto labelling) ───────────────────────
def ucs_constrained(G, Dist, Cost, start, goal, budget):
    pq = [(0.0, 0, start)]
    best = {start: [(0.0, 0)]}
    parent = {(start, 0): None}
    closed = set()

    while pq:
        d, e, node = heapq.heappop(pq)
        if (node, e) in closed:
            continue
        closed.add((node, e))

        if node == goal:
            path, state = [], (goal, e)
            while state is not None:
                path.append(state[0])
                state = parent[state]
            return d, e, path[::-1], len(closed)

        for nb in G.get(node, []):
            edge = f"{node},{nb}"
            if edge not in Dist or edge not in Cost:
                continue
            nd = d + Dist[edge]
            ne = e + Cost[edge]
            if ne > budget:
                continue
            nb_labels = best.get(nb, [])
            if any(ld <= nd and le <= ne for ld, le in nb_labels):
                continue
            best[nb] = [(ld, le) for ld, le in nb_labels if not (nd <= ld and ne <= le)]
            best[nb].append((nd, ne))
            parent[(nb, ne)] = (node, e)
            heapq.heappush(pq, (nd, ne, nb))

    return float("inf"), -1, [], len(closed)


# ── Heuristics ─────────────────────────────────────────────────────────────────
def haversine_heuristic(Coord, goal):
    """Straight-line Haversine distance from every node to goal, in decimetres."""
    glon, glat = Coord[goal][0] / 1e6, Coord[goal][1] / 1e6
    φ2, λ2 = math.radians(glat), math.radians(glon)
    h = {}
    for node, (rlon, rlat) in Coord.items():
        φ1 = math.radians(rlat / 1e6)
        λ1 = math.radians(rlon / 1e6)
        a = math.sin((φ2 - φ1) / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin((λ2 - λ1) / 2) ** 2
        h[node] = 2 * R_DM * math.asin(math.sqrt(a))
    return h


def pythagorean_heuristic(Coord, goal):
    """Flat-earth Euclidean distance from every node to goal, in decimetres."""
    glon, glat = Coord[goal][0] / 1e6, Coord[goal][1] / 1e6
    h = {}
    for node, (rlon, rlat) in Coord.items():
        dlat = glat - rlat / 1e6
        dlon = glon - rlon / 1e6
        h[node] = math.sqrt(dlat ** 2 + dlon ** 2) * 1_000_000
    return h


# ── A* (energy-constrained, Pareto labelling) ────────────────────────────────
def astar_constrained(G, Dist, Cost, start, goal, budget, h):
    pq = [(h.get(start, 0.0), 0.0, 0, start)]
    best = {start: [(0.0, 0)]}
    parent = {(start, 0): None}
    closed = set()

    while pq:
        f, d, e, node = heapq.heappop(pq)
        if (node, e) in closed:
            continue
        closed.add((node, e))

        if node == goal:
            path, state = [], (goal, e)
            while state is not None:
                path.append(state[0])
                state = parent[state]
            return d, e, path[::-1], len(closed)

        for nb in G.get(node, []):
            edge = f"{node},{nb}"
            if edge not in Dist or edge not in Cost:
                continue
            nd = d + Dist[edge]
            ne = e + Cost[edge]
            if ne > budget:
                continue
            nb_labels = best.get(nb, [])
            if any(ld <= nd and le <= ne for ld, le in nb_labels):
                continue
            best[nb] = [(ld, le) for ld, le in nb_labels if not (nd <= ld and ne <= le)]
            best[nb].append((nd, ne))
            parent[(nb, ne)] = (node, e)
            heapq.heappush(pq, (nd + h.get(nb, 0.0), nd, ne, nb))

    return float("inf"), -1, [], len(closed)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    base_dir = Path(__file__).resolve().parent
    G, Coord, Dist, Cost, nodes = generate_graph()

    # Save generated data to JSON (mirrors original part1 structure)
    with (base_dir / "G.json").open("w")     as f: json.dump(G,     f, indent=2)
    with (base_dir / "Coord.json").open("w") as f: json.dump(Coord,  f, indent=2)
    with (base_dir / "Dist.json").open("w")  as f: json.dump(Dist,   f, indent=2)
    with (base_dir / "Cost.json").open("w")  as f: json.dump(Cost,   f, indent=2)

    n_edges = sum(len(v) for v in G.values())
    print(f"Generated graph: {N_NODES} nodes, {n_edges} directed edges  (seed={SEED})")
    print(f"Start: {START}  →  Goal: {GOAL}")

    # ── Task 1 ────────────────────────────────────────────────────────────────
    t1_dist, t1_path, t1_states = ucs(G, Dist, START, GOAL)
    print("\n" + "=" * 60)
    print("Task 1: UCS (no energy constraint)")
    print(f"  Distance : {t1_dist:.2f} dm")
    print(f"  Path     : {'->'.join(t1_path)}")
    print(f"  Nodes    : {len(t1_path)}")
    print(f"  States   : {t1_states}")

    # Set budget: energy of the UCS-optimal path × 1.3  (tight but always feasible)
    t1_energy = sum(Cost.get(f"{t1_path[i]},{t1_path[i+1]}", 0)
                    for i in range(len(t1_path) - 1))
    BUDGET = int(t1_energy * 1.3)
    print(f"\n  Energy along UCS path : {t1_energy}")
    print(f"  Budget set to         : {BUDGET}  (= {t1_energy} × 1.3)")

    # ── Task 2 ────────────────────────────────────────────────────────────────
    t2_dist, t2_energy, t2_path, t2_states = ucs_constrained(
        G, Dist, Cost, START, GOAL, BUDGET)
    print("\n" + "=" * 60)
    print("Task 2: UCS (energy-constrained)")
    if t2_path:
        print(f"  Distance : {t2_dist:.2f} dm")
        print(f"  Energy   : {t2_energy}")
        print(f"  Path     : {'->'.join(t2_path)}")
        print(f"  Nodes    : {len(t2_path)}")
        print(f"  States   : {t2_states}")
    else:
        print("  No feasible path found within budget.")

    # ── Heuristics ────────────────────────────────────────────────────────────
    h_hav  = haversine_heuristic(Coord, GOAL)
    h_pyth = pythagorean_heuristic(Coord, GOAL)

    # ── Task 3a: A* Haversine ─────────────────────────────────────────────────
    t3a_dist, t3a_energy, t3a_path, t3a_states = astar_constrained(
        G, Dist, Cost, START, GOAL, BUDGET, h_hav)
    print("\n" + "=" * 60)
    print("Task 3a: A* — Haversine heuristic")
    if t3a_path:
        print(f"  Distance : {t3a_dist:.2f} dm")
        print(f"  Energy   : {t3a_energy}")
        print(f"  Path     : {'->'.join(t3a_path)}")
        print(f"  Nodes    : {len(t3a_path)}")
        print(f"  States   : {t3a_states}")
    else:
        print("  No feasible path found within budget.")

    # ── Task 3b: A* Pythagorean ───────────────────────────────────────────────
    t3b_dist, t3b_energy, t3b_path, t3b_states = astar_constrained(
        G, Dist, Cost, START, GOAL, BUDGET, h_pyth)
    print("\n" + "=" * 60)
    print("Task 3b: A* — Pythagorean heuristic")
    if t3b_path:
        print(f"  Distance : {t3b_dist:.2f} dm")
        print(f"  Energy   : {t3b_energy}")
        print(f"  Path     : {'->'.join(t3b_path)}")
        print(f"  Nodes    : {len(t3b_path)}")
        print(f"  States   : {t3b_states}")
    else:
        print("  No feasible path found within budget.")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Comparison")
    print(f"{'Algorithm':<32} {'States':>8} {'Reduction':>11} {'Optimality':>12}")
    print("-" * 65)
    if t2_path:
        print(f"{'UCS constrained (baseline)':<32} {t2_states:>8} {'--':>11} {'100.00%':>12}")
    if t3a_path and t2_path:
        red_a = (t2_states - t3a_states) / t2_states * 100
        opt_a = t2_dist / t3a_dist * 100
        print(f"{'A* Haversine':<32} {t3a_states:>8} {red_a:>10.1f}% {opt_a:>11.2f}%")
    if t3b_path and t2_path:
        red_b = (t2_states - t3b_states) / t2_states * 100
        opt_b = t2_dist / t3b_dist * 100
        print(f"{'A* Pythagorean':<32} {t3b_states:>8} {red_b:>10.1f}% {opt_b:>11.2f}%")


if __name__ == "__main__":
    main()
