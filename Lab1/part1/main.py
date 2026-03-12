import heapq
import json
import math
import matplotlib.pyplot as plt
from pathlib import Path

# ── Fixed problem parameters ───────────────────────────────────────────────────
START = "1"
GOAL = "50"
ENERGY_BUDGET = 287932
MAX_PENALTY = 2     # cap for heuristic scaling parameter to prevent extreme values
WARMUP_STATES = 100  # EA agents use h_para=1 for the first this many expansions;
                    # from state WARMUP_STATES onwards a,b drive h_para

# ── Data loading ───────────────────────────────────────────────────────────────
def load_data(base_dir: Path):
	with (base_dir / "G.json").open("r", encoding="utf-8") as f:
		G = json.load(f)
	with (base_dir / "Coord.json").open("r", encoding="utf-8") as f:
		Coord = json.load(f)
	with (base_dir / "Dist.json").open("r", encoding="utf-8") as f:
		Dist = json.load(f)
	with (base_dir / "Cost.json").open("r", encoding="utf-8") as f:
		Cost = json.load(f)
	return G, Coord, Dist, Cost


# ── Task 1: UCS (no energy constraint) ────────────────────────────────────
def ucs(G, Dist, start, goal=None):
	# Standard UCS — minimise total distance, no energy constraint.

	pq = [(0.0, start)]
	best = {start: 0.0}
	parent = {start: None}

	visited_states = 0
	while pq:
		d, node = heapq.heappop(pq)
		if d > best[node]:
			continue
		visited_states += 1
		if node == goal:
			break
		for nb in G.get(node, []):
			edge = f"{node},{nb}"
			if edge not in Dist:
				continue
			nd = d + float(Dist[edge])
			if nd < best.get(nb, float("inf")):
				best[nb] = nd
				parent[nb] = node
				heapq.heappush(pq, (nd, nb))

	if goal is None:
		return best  # full distance map for heuristic use

	if goal not in best:
		return float("inf"), [], visited_states

	path, cur = [], goal
	while cur is not None:
		path.append(cur)
		cur = parent[cur]
	return best[goal], path[::-1], visited_states


# ── Task 2: UCS with energy constraint ────────────────────────────────────────
def ucs_constrained(G, Dist, Cost, start, goal, budget):
	"""
	UCS (uninformed search) on expanded state (node, energy_used).

	Priority  : accumulated distance (same as UCS).
	Constraint: cumulative energy must not exceed `budget`.
	Pruning   : Pareto dominance — at each node we keep a front of
	            (dist, energy) labels.  A new label is discarded if any
	            existing label has dist <= new_dist AND energy <= new_energy.
	            Labels dominated by the new one are removed.
	"""
	# pq entries: (dist_so_far, energy_so_far, node)
	pq = [(0.0, 0, start)]

	# best[node] = list of Pareto-optimal (best energy + best dist) (dist, energy) to each nodes
	best = {start: [(0.0, 0)]}

	# parent[(node, energy)] = (parent_node, parent_energy) | None for start
	parent = {(start, 0): None}

	# meaning the optimal path to that exact (node, energy) state has been confirmed.
	# First pop for (node, energy) is always optimal under UCS.
	closed = set()

	while pq:
		d, e, node = heapq.heappop(pq)

		if (node, e) in closed:
			continue
		closed.add((node, e))

		if node == goal:
			# Reconstruct path using parent chain keyed by (node, energy).
			path, state = [], (goal, e)
			while state is not None:
				path.append(state[0])
				state = parent[state]
			return d, e, path[::-1], len(closed)

		for nb in G.get(node, []):
			edge = f"{node},{nb}"
			if edge not in Dist or edge not in Cost:
				continue
			nd = d + float(Dist[edge])
			ne = e + int(Cost[edge])

			if ne > budget:
				continue

			nb_labels = best.get(nb, [])
			# Skip if dominated by an existing label.
			if any(ld <= nd and le <= ne for ld, le in nb_labels):
				continue

			# Remove labels dominated by the new one, then add it.
			best[nb] = [(ld, le) for ld, le in nb_labels
						if not (nd <= ld and ne <= le)]
			best[nb].append((nd, ne))

			parent[(nb, ne)] = (node, e)
			heapq.heappush(pq, (nd, ne, nb))

	return float("inf"), -1, [], len(closed)


# ── Task 3a: A* with Haversine heuristic ──────────────────────────────────────
def _haversine_heuristic(Coord, goal):
	"""
	Straight-line (Haversine) distance from every node to `goal`.
	Coord values are stored as integers = degrees * 1e6 (lon, lat).
	Returns distances in decimetres (×10 from metres) to match raw Dist units.
	"""
	R = 6371000.0  # Earth radius in metres
	lon2, lat2 = (v / 1e6 for v in Coord[goal])
	lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

	h = {}
	for node, vals in Coord.items():
		lon1, lat1 = vals[0] / 1e6, vals[1] / 1e6
		lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
		dlat = lat2_r - lat1_r
		dlon = lon2_r - lon1_r
		a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
		h[node] = 2 * R * math.asin(math.sqrt(a)) * 10  # metres → decimetres
	return h


# ── Task 3b: A* with Pythagorean/Euclidean heuristic ──────────────────────────
def _pythagorean_heuristic(Coord, goal):
	"""
	Euclidean (Pythagorean) distance in lat-lon space from every node to `goal`.
	Coord values are stored as integers = degrees * 1e6 (lon, lat).
	Returns decimetres (×10 from metres) to match raw Dist units.
	"""
	METRES_PER_DEGREE = 111111.0
	lon2, lat2 = (v / 1e6 for v in Coord[goal])

	h = {}
	for node, vals in Coord.items():
		lon1, lat1 = vals[0] / 1e6, vals[1] / 1e6
		dlat = lat2 - lat1
		dlon = lon2 - lon1
		h[node] = math.sqrt(dlat**2 + dlon**2) * METRES_PER_DEGREE * 10  # metres → decimetres
	return h


def astar_constrained_haversine(G, Dist, Cost, start, goal, budget, h):
	"""
	A* on expanded state (node, energy_used) with Haversine heuristic.

	Priority  : f = g + h(node), where h is the Haversine distance.
	Constraint: cumulative energy must not exceed `budget`.
	Pruning   : Pareto-dominance labelling.
	"""
	pq = [(h.get(start, float("inf")), 0.0, 0, start)]
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
			nd = d + float(Dist[edge])
			ne = e + int(Cost[edge])

			if ne > budget:
				continue

			nb_labels = best.get(nb, [])
			if any(ld <= nd and le <= ne for ld, le in nb_labels):
				continue

			best[nb] = [(ld, le) for ld, le in nb_labels
						if not (nd <= ld and ne <= le)]
			best[nb].append((nd, ne))

			parent[(nb, ne)] = (node, e)
			nf = nd + h.get(nb, float("inf"))
			heapq.heappush(pq, (nf, nd, ne, nb))

	return float("inf"), -1, [], len(closed)


def astar_constrained_pythagorean(G, Dist, Cost, start, goal, budget, h):
	"""
	A* on expanded state (node, energy_used) with Pythagorean/Euclidean heuristic.

	Priority  : f = g + h(node), where h is the Euclidean distance.
	Constraint: cumulative energy must not exceed `budget`.
	Pruning   : Pareto-dominance labelling.
	"""
	pq = [(h.get(start, float("inf")), 0.0, 0, start)]
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
			nd = d + float(Dist[edge])
			ne = e + int(Cost[edge])

			if ne > budget:
				continue

			nb_labels = best.get(nb, [])
			if any(ld <= nd and le <= ne for ld, le in nb_labels):
				continue

			best[nb] = [(ld, le) for ld, le in nb_labels
						if not (nd <= ld and ne <= le)]
			best[nb].append((nd, ne))

			parent[(nb, ne)] = (node, e)
			nf = nd + h.get(nb, float("inf"))
			heapq.heappush(pq, (nf, nd, ne, nb))

	return float("inf"), -1, [], len(closed)


# ── Online linear regression helpers ──────────────────────────────────────────

def main():
	base_dir = Path(__file__).resolve().parent
	G, Coord, Dist, Cost = load_data(base_dir)

	# ── Task 1 ─────────────────────────────────────────────────────────────────
	t1_dist, t1_path, t1_states = ucs(G, Dist, START, GOAL)

	print("=" * 60)
	print("Task 1: UCS (relaxed — no energy constraint)")
	if t1_path:
		print(f"Shortest path: {'->'.join(t1_path)}.")
		print(f"Shortest distance: {t1_dist}.")
		print(f"Number of nodes in path: {len(t1_path)}.")
		print(f"Number of states visited: {t1_states}.")
	else:
		print("No path found.")

	# ── Task 2 ─────────────────────────────────────────────────────────────────
	t2_dist, t2_energy, t2_path, t2_states = ucs_constrained(
		G, Dist, Cost, START, GOAL, ENERGY_BUDGET
	)

	print()
	print("=" * 60)
	print("Task 2: UCS (energy-constrained shortest path)")
	if t2_path:
		print(f"Shortest path: {'->'.join(t2_path)}.")
		print(f"Shortest distance: {t2_dist}.")
		print(f"Total energy cost: {t2_energy}.")
		print(f"Number of nodes in path: {len(t2_path)}.")
		print(f"Number of states visited: {t2_states}.")
	else:
		print("No feasible path found within the energy budget.")

	# ── Task 3a: A* with Haversine ────────────────────────────────────────────
	h_haversine = _haversine_heuristic(Coord, GOAL)
	t3a_dist, t3a_energy, t3a_path, t3a_states = astar_constrained_haversine(
		G, Dist, Cost, START, GOAL, ENERGY_BUDGET, h_haversine
	)

	print()
	print("=" * 60)
	print("Task 3a: A* — Haversine heuristic")
	if t3a_path:
		print(f"Shortest path: {'->'.join(t3a_path)}.")
		print(f"Shortest distance: {t3a_dist}.")
		print(f"Total energy cost: {t3a_energy}.")
		print(f"Number of nodes in path: {len(t3a_path)}.")
		print(f"Number of states visited: {t3a_states}.")
	else:
		print("No feasible path found within the energy budget.")

	# ── Task 3b: A* with Pythagorean/Euclidean ────────────────────────────────
	h_pythagorean = _pythagorean_heuristic(Coord, GOAL)
	t3b_dist, t3b_energy, t3b_path, t3b_states = astar_constrained_pythagorean(
		G, Dist, Cost, START, GOAL, ENERGY_BUDGET, h_pythagorean
	)

	print()
	print("=" * 60)
	print("Task 3b: A* — Pythagorean/Euclidean heuristic")
	if t3b_path:
		print(f"Shortest path: {'->'.join(t3b_path)}.")
		print(f"Shortest distance: {t3b_dist}.")
		print(f"Total energy cost: {t3b_energy}.")
		print(f"Number of nodes in path: {len(t3b_path)}.")
		print(f"Number of states visited: {t3b_states}.")
	else:
		print("No feasible path found within the energy budget.")

	# ── Write paths to results.txt ──────────────────────────────
	txt_path = base_dir / "results.txt"
	with txt_path.open("w", encoding="utf-8") as txt:
		def p(s=""):
			txt.write(s + "\n")

		p("Task 1: UCS (relaxed — no energy constraint)")
		p(f"Shortest path: {'->'.join(t1_path) if t1_path else 'No path found'}.")
		p()
		p("Task 2: UCS (energy-constrained shortest path)")
		p(f"Shortest path: {'->'.join(t2_path) if t2_path else 'No feasible path found within the energy budget'}.")
		p()
		p("Task 3a: A* — Haversine heuristic")
		p(f"Shortest path: {'->'.join(t3a_path) if t3a_path else 'No feasible path found within the energy budget'}.")
		p()
		p("Task 3b: A* — Pythagorean/Euclidean heuristic")
		p(f"Shortest path: {'->'.join(t3b_path) if t3b_path else 'No feasible path found within the energy budget'}.")

	# ── Write results.md (no paths) ─────────────────────────────
	out_path = base_dir / "results.md"
	with out_path.open("w", encoding="utf-8") as out:
		def w(s=""):
			out.write(s + "\n")

		w("# Heuristic Comparison: Haversine vs Pythagorean/Euclidean")
		w()

		w("## Task 1: UCS (relaxed — no energy constraint)")
		w()
		if t1_path:
			w(f"- Shortest distance: {t1_dist:.5f}")
			w(f"- Number of nodes in path: {len(t1_path)}")
			w(f"- Number of states visited: {t1_states}")
		else:
			w("No path found.")

		w()
		w("## Task 2: UCS (energy-constrained shortest path)")
		w()
		if t2_path:
			w(f"- Shortest distance: {t2_dist:.5f}")
			w(f"- Total energy cost: {t2_energy}")
			w(f"- Number of nodes in path: {len(t2_path)}")
			w(f"- Number of states visited: {t2_states}")
		else:
			w("No feasible path found within the energy budget.")

		w()
		w("## Task 3a: A* — Haversine heuristic (great-circle distance)")
		w()
		if t3a_path:
			w(f"- Shortest distance: {t3a_dist:.5f}")
			w(f"- Total energy cost: {t3a_energy}")
			w(f"- Number of nodes in path: {len(t3a_path)}")
			w(f"- Number of states visited: {t3a_states}")
		else:
			w("No feasible path found within the energy budget.")

		w()
		w("## Task 3b: A* — Pythagorean/Euclidean heuristic")
		w()
		if t3b_path:
			w(f"- Shortest distance: {t3b_dist:.5f}")
			w(f"- Total energy cost: {t3b_energy}")
			w(f"- Number of nodes in path: {len(t3b_path)}")
			w(f"- Number of states visited: {t3b_states}")
		else:
			w("No feasible path found within the energy budget.")

		w()
		w("## Comparison: states visited and path accuracy")
		w()
		hav_reduction     = (t2_states - t3a_states) / t2_states * 100
		pyth_reduction    = (t2_states - t3b_states) / t2_states * 100
		t3a_accuracy  = t2_dist / t3a_dist * 100 if t3a_dist else 0.0
		t3b_accuracy  = t2_dist / t3b_dist * 100 if t3b_dist else 0.0
		w(f"| Algorithm                                      | States visited | Reduction vs UCS | Path optimality    |")
		w(f"|------------------------------------------------|----------------|------------------|--------------------|")
		w(f"| Task 2  UCS constrained (optimal)              | {t2_states:>14} | --               | 100.00% (baseline) |")
		w(f"| Task 3a A* Haversine                           | {t3a_states:>14} | {hav_reduction:>7.1f}%          | {t3a_accuracy:.2f}%             |")
		w(f"| Task 3b A* Pythagorean                         | {t3b_states:>14} | {pyth_reduction:>7.1f}%          | {t3b_accuracy:.2f}%             |")


if __name__ == "__main__":
	main()
