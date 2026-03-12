import heapq
import json
import math
import numpy as np
from pathlib import Path

# ── Fixed problem parameters ───────────────────────────────────────────────────
START = "1"
GOAL = "50"
ENERGY_BUDGET = 287932


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
	Returns distances in metres — same unit as Dist — so admissible.
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
		h[node] = 2 * R * math.asin(math.sqrt(a))
	return h


# ── Task 3b: A* with Pythagorean/Euclidean heuristic ──────────────────────────
def _pythagorean_heuristic(Coord, goal):
	"""
	Euclidean (Pythagorean) distance in lat-lon space from every node to `goal`.
	Coord values are stored as integers = degrees * 1e6 (lon, lat).
	Uses simple Euclidean: sqrt(dlat^2 + dlon^2) scaled to approximate metres.
	
	Approximation: 1 degree ≈ 111,111 metres (mean Earth radius conversion)
	"""
	METRES_PER_DEGREE = 111111.0
	lon2, lat2 = (v / 1e6 for v in Coord[goal])

	h = {}
	for node, vals in Coord.items():
		lon1, lat1 = vals[0] / 1e6, vals[1] / 1e6
		dlat = lat2 - lat1
		dlon = lon2 - lon1
		# Euclidean distance in degrees, scaled to metres
		h[node] = math.sqrt(dlat**2 + dlon**2) * METRES_PER_DEGREE
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


# ── Per-edge geometric distance helpers for energy-aware regression ────────────
def _edge_distances_haversine(Coord, Cost):
	"""Return (dists, costs) arrays using Haversine distance per edge endpoint pair."""
	R = 6371000.0
	dists, costs = [], []
	for edge, cost in Cost.items():
		u, v = edge.split(",", 1)
		if u not in Coord or v not in Coord:
			continue
		lon1, lat1 = Coord[u][0] / 1e6, Coord[u][1] / 1e6
		lon2, lat2 = Coord[v][0] / 1e6, Coord[v][1] / 1e6
		lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
		lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
		dlat = lat2_r - lat1_r
		dlon = lon2_r - lon1_r
		a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
		dists.append(2 * R * math.asin(math.sqrt(a)))
		costs.append(int(cost))
	return np.array(dists), np.array(costs)


def _edge_distances_pythagorean(Coord, Cost):
	"""Return (dists, costs) arrays using Pythagorean distance per edge endpoint pair."""
	METRES_PER_DEGREE = 111111.0
	dists, costs = [], []
	for edge, cost in Cost.items():
		u, v = edge.split(",", 1)
		if u not in Coord or v not in Coord:
			continue
		lon1, lat1 = Coord[u][0] / 1e6, Coord[u][1] / 1e6
		lon2, lat2 = Coord[v][0] / 1e6, Coord[v][1] / 1e6
		dlat = lat2 - lat1
		dlon = lon2 - lon1
		dists.append(math.sqrt(dlat ** 2 + dlon ** 2) * METRES_PER_DEGREE)
		costs.append(int(cost))
	return np.array(dists), np.array(costs)


def astar_constrained_haversine_energyaware(G, Dist, Cost, Coord, start, goal, budget, h):
	"""
	A* with Haversine heuristic AND admissible energy-aware scaling.

	Regresses cost against per-edge HAVERSINE distances: cost ≈ a * haversine_dist + b
	(a and b are specific to Haversine geometry, not road distances)

	Priority  : f = g + h(n) * (B / (B - (a*h(n) + b)))
	"""
	# Regress cost against per-edge Haversine distances (same metric as heuristic)
	dists, costs = _edge_distances_haversine(Coord, Cost)

	# Linear regression
	A_mat = np.vstack([dists, np.ones(len(dists))]).T
	a, b = np.linalg.lstsq(A_mat, costs, rcond=None)[0]

	# Pearson correlation
	corr = np.corrcoef(dists, costs)[0, 1]
	
	h_start = h.get(start, float("inf"))
	
	pq = [(h_start, 0.0, 0, start)]
	best = {start: [(0.0, 0)]}
	parent = {(start, 0): None}
	closed = set()
	linearity_info = (a, b, corr)

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
			return d, e, path[::-1], len(closed), linearity_info

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
			
			# Admissible energy-aware heuristic with Pythagorean base: h(n) * (B / (B - (a*h(n) + b)))
			h_nb = h.get(nb, float("inf"))
			estimated_cost = a * h_nb + b
			remaining_energy_budget = budget - ne
			estimated_after_nb = remaining_energy_budget - estimated_cost
			h_para = remaining_energy_budget / estimated_after_nb if estimated_after_nb > 0 else 10.0
			
			# Scale heuristic by budget ratio (higher when less budget remains)
			if remaining_energy_budget > estimated_cost:
				nf = nd + h_nb * h_para

			
			heapq.heappush(pq, (nf, nd, ne, nb))

	return float("inf"), -1, [], len(closed), linearity_info


def astar_constrained_pythagorean_energyaware(G, Dist, Cost, Coord, start, goal, budget, h):
	"""
	A* with Pythagorean heuristic AND admissible energy-aware scaling.

	Regresses cost against per-edge PYTHAGOREAN distances: cost ≈ a * pythagorean_dist + b
	(a and b are specific to Pythagorean geometry — different from Haversine variant)

	Priority  : f = g + h(n) * (B / (B - (a*h(n) + b)))
	"""
	# Regress cost against per-edge Pythagorean distances (same metric as heuristic)
	dists, costs = _edge_distances_pythagorean(Coord, Cost)

	# Linear regression
	A_mat = np.vstack([dists, np.ones(len(dists))]).T
	a, b = np.linalg.lstsq(A_mat, costs, rcond=None)[0]

	# Pearson correlation
	corr = np.corrcoef(dists, costs)[0, 1]

	h_start = h.get(start, float("inf"))

	pq = [(h_start, 0.0, 0, start)]
	best = {start: [(0.0, 0)]}
	parent = {(start, 0): None}
	closed = set()
	linearity_info = (a, b, corr)

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
			return d, e, path[::-1], len(closed), linearity_info

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

			h_nb = h.get(nb, float("inf"))
			estimated_cost = a * h_nb + b
			remaining_energy_budget = budget - ne
			estimated_after_nb = remaining_energy_budget - estimated_cost
			h_para = remaining_energy_budget / estimated_after_nb if estimated_after_nb > 0 else 10.0
			
			# Scale heuristic by budget ratio (higher when less budget remains)
			if remaining_energy_budget > estimated_cost:
				nf = nd + h_nb * h_para

			heapq.heappush(pq, (nf, nd, ne, nb))

	return float("inf"), -1, [], len(closed), linearity_info


# ── Entry point ────────────────────────────────────────────────────────────────
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

	# ── Task 3c: A* with Haversine + energy-aware ────────────────────────────
	t3c_dist, t3c_energy, t3c_path, t3c_states, t3c_lin = astar_constrained_haversine_energyaware(
		G, Dist, Cost, Coord, START, GOAL, ENERGY_BUDGET, h_haversine
	)
	a_hav, b_hav, corr_hav = t3c_lin

	print()
	print("=" * 60)
	print("Task 3c: A* — Haversine + energy-aware h(n)*B/(B-(a*h(n)+b))")
	print(f"  Linearity (Haversine): cost ≈ {a_hav:.6f} * haversine_dist + {b_hav:.2f}")
	print(f"  Pearson correlation: {corr_hav:.6f}")
	if t3c_path:
		print(f"Shortest path: {'->'.join(t3c_path)}.")
		print(f"Shortest distance: {t3c_dist}.")
		print(f"Total energy cost: {t3c_energy}.")
		print(f"Number of nodes in path: {len(t3c_path)}.")
		print(f"Number of states visited: {t3c_states}.")
	else:
		print("No feasible path found within the energy budget.")

	# ── Task 3d: A* with Pythagorean + energy-aware ───────────────────────────
	t3d_dist, t3d_energy, t3d_path, t3d_states, t3d_lin = astar_constrained_pythagorean_energyaware(
		G, Dist, Cost, Coord, START, GOAL, ENERGY_BUDGET, h_pythagorean
	)
	a_pyth, b_pyth, corr_pyth = t3d_lin

	print()
	print("=" * 60)
	print("Task 3d: A* — Pythagorean + energy-aware h(n)*B/(B-(a*h(n)+b))")
	print(f"  Linearity (Pythagorean): cost ≈ {a_pyth:.6f} * pythagorean_dist + {b_pyth:.2f}")
	print(f"  Pearson correlation: {corr_pyth:.6f}")
	if t3d_path:
		print(f"Shortest path: {'->'.join(t3d_path)}.")
		print(f"Shortest distance: {t3d_dist}.")
		print(f"Total energy cost: {t3d_energy}.")
		print(f"Number of nodes in path: {len(t3d_path)}.")
		print(f"Number of states visited: {t3d_states}.")
	else:
		print("No feasible path found within the energy budget.")

	# ── Write paths to results-v2.txt ──────────────────────────────
	txt_path = base_dir / "results-v2.txt"
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
		p()
		p("Task 3c: A* — Haversine + energy-aware")
		p(f"Shortest path: {'->'.join(t3c_path) if t3c_path else 'No feasible path found within the energy budget'}.")
		p()
		p("Task 3d: A* — Pythagorean + energy-aware")
		p(f"Shortest path: {'->'.join(t3d_path) if t3d_path else 'No feasible path found within the energy budget'}.")

	# ── Write results-v2.md (no paths) ─────────────────────────────
	out_path = base_dir / "results-v2.md"
	with out_path.open("w", encoding="utf-8") as out:
		def w(s=""):
			out.write(s + "\n")

		w("# Heuristic Comparison: Haversine vs Pythagorean/Euclidean")
		w()

		w("## Task 1: UCS (relaxed — no energy constraint)")
		w()
		if t1_path:
			w(f"- Shortest distance: {t1_dist:.5f} m")
			w(f"- Number of nodes in path: {len(t1_path)}")
			w(f"- Number of states visited: {t1_states}")
		else:
			w("No path found.")

		w()
		w("## Task 2: UCS (energy-constrained shortest path)")
		w()
		if t2_path:
			w(f"- Shortest distance: {t2_dist:.5f} m")
			w(f"- Total energy cost: {t2_energy}")
			w(f"- Number of nodes in path: {len(t2_path)}")
			w(f"- Number of states visited: {t2_states}")
		else:
			w("No feasible path found within the energy budget.")

		w()
		w("## Task 3a: A* — Haversine heuristic (great-circle distance)")
		w()
		if t3a_path:
			w(f"- Shortest distance: {t3a_dist:.5f} m")
			w(f"- Total energy cost: {t3a_energy}")
			w(f"- Number of nodes in path: {len(t3a_path)}")
			w(f"- Number of states visited: {t3a_states}")
		else:
			w("No feasible path found within the energy budget.")

		w()
		w("## Task 3b: A* — Pythagorean/Euclidean heuristic")
		w()
		w("Formula: `sqrt(dlat^2 + dlon^2) * 111111 m/degree`")
		w()
		if t3b_path:
			w(f"- Shortest distance: {t3b_dist:.5f} m")
			w(f"- Total energy cost: {t3b_energy}")
			w(f"- Number of nodes in path: {len(t3b_path)}")
			w(f"- Number of states visited: {t3b_states}")
		else:
			w("No feasible path found within the energy budget.")

		w()
		w("## Task 3c: A* — Haversine + energy-aware `h(n) * B / (B - (a*h(n) + b))`")
		w()
		w(f"- Linearity (Haversine): cost ≈ {a_hav:.6f} × haversine_dist + {b_hav:.2f}")
		w(f"- Pearson correlation: {corr_hav:.6f}")
		w()
		if t3c_path:
			w(f"- Shortest distance: {t3c_dist:.5f} m")
			w(f"- Total energy cost: {t3c_energy}")
			w(f"- Number of nodes in path: {len(t3c_path)}")
			w(f"- Number of states visited: {t3c_states}")
		else:
			w("No feasible path found within the energy budget.")

		w()
		w("## Task 3d: A* — Pythagorean + energy-aware `h(n) * B / (B - (a*h(n) + b))`")
		w()
		w(f"- Linearity (Pythagorean): cost ≈ {a_pyth:.6f} × pythagorean_dist + {b_pyth:.2f}")
		w(f"- Pearson correlation: {corr_pyth:.6f}")
		w()
		if t3d_path:
			w(f"- Shortest distance: {t3d_dist:.5f} m")
			w(f"- Total energy cost: {t3d_energy}")
			w(f"- Number of nodes in path: {len(t3d_path)}")
			w(f"- Number of states visited: {t3d_states}")
		else:
			w("No feasible path found within the energy budget.")

		w()
		w("## Comparison: states visited and path accuracy")
		w()
		hav_reduction    = (t2_states - t3a_states) / t2_states * 100
		pyth_reduction   = (t2_states - t3b_states) / t2_states * 100
		hav_ea_reduction = (t2_states - t3c_states) / t2_states * 100
		pyth_ea_reduction= (t2_states - t3d_states) / t2_states * 100
		t3a_accuracy = t2_dist / t3a_dist * 100 if t3a_dist else 0.0
		t3b_accuracy = t2_dist / t3b_dist * 100 if t3b_dist else 0.0
		t3c_accuracy = t2_dist / t3c_dist * 100 if t3c_dist else 0.0
		t3d_accuracy = t2_dist / t3d_dist * 100 if t3d_dist else 0.0
		w(f"| Algorithm                             | States visited | States Visited Reduction | Path optimality |")
		w(f"|---------------------------------------|----------------|------------------|----------------------------|")
		w(f"| Task 2 UCS constrained (optimal)      | {t2_states:>14} | --               | 100.00% (baseline)         |")
		w(f"| Task 3a A* Haversine                  | {t3a_states:>14} | {hav_reduction:>7.1f}%          | {t3a_accuracy:.2f}%                       |")
		w(f"| Task 3b A* Pythagorean                | {t3b_states:>14} | {pyth_reduction:>7.1f}%          | {t3b_accuracy:.2f}%                       |")
		w(f"| Task 3c A* Haversine + energy-aware   | {t3c_states:>14} | {hav_ea_reduction:>7.1f}%          | {t3c_accuracy:.2f}%                       |")
		w(f"| Task 3d A* Pythagorean + energy-aware | {t3d_states:>14} | {pyth_ea_reduction:>7.1f}%          | {t3d_accuracy:.2f}%                       |")


if __name__ == "__main__":
	main()
