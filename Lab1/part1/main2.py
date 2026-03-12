import heapq
import json
import math
import matplotlib.pyplot as plt
from pathlib import Path

# ── Fixed problem parameters ───────────────────────────────────────────────────
START = "1"
GOAL = "50"
ENERGY_BUDGET = 287932
MAX_PENALTY = 10     # cap for heuristic scaling parameter to prevent extreme values
WARMUP_STATES = 10  # EA agents use h_para=1 for the first this many expansions;
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
	Returns distances in metres. Note: Dist stores raw decimetres (~10× metres),
	so this heuristic underestimates by ~10× — admissible, but weak.
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

	Approximation: 1 degree ≈ 111,111 metres (mean Earth radius conversion).
	Returns metres. Note: Dist stores raw decimetres (~10× metres),
	so this heuristic underestimates by ~10× — admissible, but weak.
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


# ── Online linear regression helpers ──────────────────────────────────────────
def _online_reg_update(n, sx, sy, sxx, sxy, syy, x, y):
	"""Add one data point (x, y) to running sums; return updated sums and (a, b)."""
	n   += 1
	sx  += x;  sy  += y
	sxx += x * x;  sxy += x * y;  syy += y * y
	if n >= 2:
		denom = n * sxx - sx * sx
		a = (n * sxy - sx * sy) / denom if denom != 0.0 else 0.0
		b = (sy - a * sx) / n
	else:
		a, b = 0.0, 0.0
	return n, sx, sy, sxx, sxy, syy, a, b


def _pearson_from_sums(n, sx, sy, sxx, sxy, syy):
	"""Pearson r from running sums."""
	if n < 2:
		return 0.0
	denom = math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
	return (n * sxy - sx * sy) / denom if denom != 0.0 else 0.0


def _plot_ea_history(history, title, save_path):
	"""
	Save a 4-panel diagram for an EA agent:
	  Panel 1 – a  (y-axis clipped to 2–98th percentile; full curve inset)
	  Panel 2 – b  (y-axis clipped to 2–98th percentile; full curve inset)
	  Panel 3 – h_para (full; late thrashing explained by rem → 0, not bad a,b)
	  Panel 4 – remaining budget at each expansion (shows WHY h_para spikes late)

	history  : list of (step, a, b, h_para_rep, rem) recorded once per expansion.
	save_path: pathlib.Path — where to write the PNG.
	A red dashed vertical line marks the end of the warm-up period.
	"""
	if not history:
		return
	import numpy as np
	steps   = [r[0] for r in history]
	a_vals  = [r[1] for r in history]
	b_vals  = [r[2] for r in history]
	hp_vals = [r[3] for r in history]
	rem_vals= [r[4] for r in history]

	def clipped_ylim(vals, lo=2, hi=98, pad=0.15):
		"""Return (ymin, ymax) clipped to percentile range with padding."""
		vlo, vhi = float(np.percentile(vals, lo)), float(np.percentile(vals, hi))
		margin = (vhi - vlo) * pad if vhi != vlo else abs(vhi) * 0.1 + 1
		return vlo - margin, vhi + margin

	fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
	fig.suptitle(title, fontsize=11)

	# Panel 1 – a
	axes[0].plot(steps, a_vals, lw=0.8, color="steelblue")
	axes[0].set_ylabel("a")
	axes[0].set_ylim(*clipped_ylim(a_vals))
	axes[0].axvline(WARMUP_STATES, color="red", lw=1.5, linestyle="--",
	               label=f"warmup ends (state {WARMUP_STATES})")
	axes[0].legend(fontsize=8)

	# Panel 2 – b (clipped — initial OLS with 1-2 points gives extreme transients)
	axes[1].plot(steps, b_vals, lw=0.8, color="darkorange")
	axes[1].set_ylabel("b\n(y clipped 2–98%)")
	axes[1].set_ylim(*clipped_ylim(b_vals))
	axes[1].axvline(WARMUP_STATES, color="red", lw=1.5, linestyle="--")

	# Panel 3 – h_para (full range; late spikes are expected — see panel 4)
	axes[2].plot(steps, hp_vals, lw=0.8, color="mediumseagreen")
	axes[2].set_ylabel("h_para")
	axes[2].axvline(WARMUP_STATES, color="red", lw=1.5, linestyle="--")
	axes[2].annotate(
		"late spikes: rem shrinks\n→ rem/(rem−(a·h+b)) blows up\n(a,b are stable — see above)",
		xy=(steps[int(len(steps) * 0.72)], float(np.percentile(hp_vals, 98))),
		xytext=(steps[int(len(steps) * 0.45)], float(np.percentile(hp_vals, 99)) * 0.85),
		fontsize=7, color="darkgreen",
		arrowprops=dict(arrowstyle="->", color="darkgreen", lw=0.8),
	)

	# Panel 4 – remaining budget (explains WHY h_para spikes when rem → 0)
	axes[3].plot(steps, rem_vals, lw=0.8, color="slategrey")
	axes[3].set_ylabel("remaining\nbudget")
	axes[3].axvline(WARMUP_STATES, color="red", lw=1.5, linestyle="--")
	axes[3].set_xlabel("expansion step (state no.)")

	plt.tight_layout()
	plt.savefig(save_path, dpi=150)
	plt.close(fig)
	print(f"  [diagram saved → {save_path.name}]")


def astar_constrained_haversine_energyaware(G, Dist, Cost, Coord, start, goal, budget, h):
	"""
	A* + Haversine heuristic + online energy-aware scaling (all-neighbours variant).

	Data collection: at every expanded node, (haversine_edge_dist, edge_cost)
	is gathered for ALL valid neighbours — i.e. regression uses the Haversine
	geometric estimate of each individual edge (node→nb), NOT h[nb] which is
	distance-to-goal.
	Warmup and history tracking same as other EA variants.
	"""
	n_data = 0
	sx = sy = sxx = sxy = syy = 0.0
	a, b = 0.0, 0.0
	history = []

	pq = [(h.get(start, float("inf")), 0.0, 0, start)]
	best   = {start: [(0.0, 0)]}
	parent = {(start, 0): None}
	closed = set()

	while pq:
		f, d, e, node = heapq.heappop(pq)

		if (node, e) in closed:
			continue
		closed.add((node, e))
		step = len(closed)

		if node == goal:
			path, state = [], (goal, e)
			while state is not None:
				path.append(state[0])
				state = parent[state]
			h_cur     = h.get(node, 0.0)
			rem_cur   = budget - e
			after_cur = rem_cur - (a * h_cur + b)
			if step >= WARMUP_STATES and after_cur > 0:
				h_para_rep = min(rem_cur / after_cur, float(MAX_PENALTY))
			elif step >= WARMUP_STATES:
				h_para_rep = float(MAX_PENALTY)
			else:
				h_para_rep = 1.0
			history.append((step, a, b, h_para_rep, rem_cur))
			corr = _pearson_from_sums(n_data, sx, sy, sxx, sxy, syy)
			return d, e, path[::-1], len(closed), (a, b, corr), history

		# ── Regression update: ALL valid neighbours (Haversine edge estimate) ──
		R_hav = 6371000.0
		_lon1c, _lat1c = Coord[node][0] / 1e6, Coord[node][1] / 1e6
		_lat1r, _lon1r = math.radians(_lat1c), math.radians(_lon1c)
		for nb in G.get(node, []):
			edge = f"{node},{nb}"
			if edge not in Cost:
				continue
			if nb not in Coord:
				continue
			_lon2c, _lat2c = Coord[nb][0] / 1e6, Coord[nb][1] / 1e6
			_lat2r, _lon2r = math.radians(_lat2c), math.radians(_lon2c)
			_dlat = _lat2r - _lat1r
			_dlon = _lon2r - _lon1r
			_sinA = math.sin(_dlat / 2) ** 2 + math.cos(_lat1r) * math.cos(_lat2r) * math.sin(_dlon / 2) ** 2
			hav_edge = 2 * R_hav * math.asin(math.sqrt(_sinA))
			n_data, sx, sy, sxx, sxy, syy, a, b = _online_reg_update(
				n_data, sx, sy, sxx, sxy, syy, hav_edge, float(Cost[edge])
			)

		# ── History record (after regression, before expansion) ────────────────
		h_cur     = h.get(node, 0.0)
		rem_cur   = budget - e
		after_cur = rem_cur - (a * h_cur + b)
		if step >= WARMUP_STATES and after_cur > 0:
			h_para_rep = min(rem_cur / after_cur, float(MAX_PENALTY))
		elif step >= WARMUP_STATES:
			h_para_rep = float(MAX_PENALTY)
		else:
			h_para_rep = 1.0
		history.append((step, a, b, h_para_rep, rem_cur))

		# ── Search expansion ───────────────────────────────────────────────────
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
			best[nb] = [(ld, le) for ld, le in nb_labels if not (nd <= ld and ne <= le)]
			best[nb].append((nd, ne))
			parent[(nb, ne)] = (node, e)

			h_nb  = h.get(nb, float("inf"))
			rem   = budget - ne
			after = rem - (a * h_nb + b)
			if step >= WARMUP_STATES and after > 0:
				h_para = min(rem / after, float(MAX_PENALTY))
			else:
				h_para = 1.0
			heapq.heappush(pq, (nd + h_nb * h_para, nd, ne, nb))

	corr = _pearson_from_sums(n_data, sx, sy, sxx, sxy, syy)
	return float("inf"), -1, [], len(closed), (a, b, corr), history


def astar_constrained_pythagorean_energyaware(G, Dist, Cost, Coord, start, goal, budget, h):
	"""
	A* + Pythagorean heuristic + online energy-aware scaling (all-neighbours variant).

	Data collection: at every expanded node, (pythagorean_edge_dist, edge_cost)
	is gathered for ALL valid neighbours — i.e. regression uses the Pythagorean
	geometric estimate of each individual edge (node→nb), NOT h[nb].
	Warmup and history tracking same as other EA variants.
	"""
	n_data = 0
	sx = sy = sxx = sxy = syy = 0.0
	a, b = 0.0, 0.0
	history = []

	pq = [(h.get(start, float("inf")), 0.0, 0, start)]
	best   = {start: [(0.0, 0)]}
	parent = {(start, 0): None}
	closed = set()

	while pq:
		f, d, e, node = heapq.heappop(pq)

		if (node, e) in closed:
			continue
		closed.add((node, e))
		step = len(closed)

		if node == goal:
			path, state = [], (goal, e)
			while state is not None:
				path.append(state[0])
				state = parent[state]
			h_cur     = h.get(node, 0.0)
			rem_cur   = budget - e
			after_cur = rem_cur - (a * h_cur + b)
			if step >= WARMUP_STATES and after_cur > 0:
				h_para_rep = min(rem_cur / after_cur, float(MAX_PENALTY))
			elif step >= WARMUP_STATES:
				h_para_rep = float(MAX_PENALTY)
			else:
				h_para_rep = 1.0
			history.append((step, a, b, h_para_rep, rem_cur))
			corr = _pearson_from_sums(n_data, sx, sy, sxx, sxy, syy)
			return d, e, path[::-1], len(closed), (a, b, corr), history

		# ── Regression update: ALL valid neighbours (Pythagorean edge estimate) ─
		MPD = 111111.0
		_lon1c, _lat1c = Coord[node][0] / 1e6, Coord[node][1] / 1e6
		for nb in G.get(node, []):
			edge = f"{node},{nb}"
			if edge not in Cost:
				continue
			if nb not in Coord:
				continue
			_lon2c, _lat2c = Coord[nb][0] / 1e6, Coord[nb][1] / 1e6
			pyth_edge = math.sqrt((_lat2c - _lat1c) ** 2 + (_lon2c - _lon1c) ** 2) * MPD
			n_data, sx, sy, sxx, sxy, syy, a, b = _online_reg_update(
				n_data, sx, sy, sxx, sxy, syy, pyth_edge, float(Cost[edge])
			)

		# ── History record (after regression, before expansion) ────────────────
		h_cur     = h.get(node, 0.0)
		rem_cur   = budget - e
		after_cur = rem_cur - (a * h_cur + b)
		if step >= WARMUP_STATES and after_cur > 0:
			h_para_rep = min(rem_cur / after_cur, float(MAX_PENALTY))
		elif step >= WARMUP_STATES:
			h_para_rep = float(MAX_PENALTY)
		else:
			h_para_rep = 1.0
		history.append((step, a, b, h_para_rep, rem_cur))

		# ── Search expansion ───────────────────────────────────────────────────
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
			best[nb] = [(ld, le) for ld, le in nb_labels if not (nd <= ld and ne <= le)]
			best[nb].append((nd, ne))
			parent[(nb, ne)] = (node, e)

			h_nb  = h.get(nb, float("inf"))
			rem   = budget - ne
			after = rem - (a * h_nb + b)
			if step >= WARMUP_STATES and after > 0:
				h_para = min(rem / after, float(MAX_PENALTY))
			else:
				h_para = 1.0
			heapq.heappush(pq, (nd + h_nb * h_para, nd, ne, nb))

	corr = _pearson_from_sums(n_data, sx, sy, sxx, sxy, syy)
	return float("inf"), -1, [], len(closed), (a, b, corr), history


def astar_constrained_haversine_energyaware_realdist(G, Dist, Cost, Coord, start, goal, budget, h):
	"""
	A* + Haversine heuristic + online energy-aware scaling (real-dist variant).

	Data collection: at every expanded node, (real_road_dist[edge], edge_cost)
	is gathered for ALL valid neighbours — i.e. regression uses actual road
	distances from Dist.json, not the heuristic estimate.
	Warmup and history tracking are the same as the all-neighbours variant.
	"""
	n_data = 0
	sx = sy = sxx = sxy = syy = 0.0
	a, b = 0.0, 0.0
	history = []

	pq = [(h.get(start, float("inf")), 0.0, 0, start)]
	best   = {start: [(0.0, 0)]}
	parent = {(start, 0): None}
	closed = set()

	while pq:
		f, d, e, node = heapq.heappop(pq)

		if (node, e) in closed:
			continue
		closed.add((node, e))
		step = len(closed)

		if node == goal:
			path, state = [], (goal, e)
			while state is not None:
				path.append(state[0])
				state = parent[state]
			h_cur     = h.get(node, 0.0)
			rem_cur   = budget - e
			after_cur = rem_cur - (a * h_cur + b)
			if step >= WARMUP_STATES and after_cur > 0:
				h_para_rep = min(rem_cur / after_cur, float(MAX_PENALTY))
			elif step >= WARMUP_STATES:
				h_para_rep = float(MAX_PENALTY)
			else:
				h_para_rep = 1.0
			history.append((step, a, b, h_para_rep, rem_cur))
			corr = _pearson_from_sums(n_data, sx, sy, sxx, sxy, syy)
			return d, e, path[::-1], len(closed), (a, b, corr), history

		# ── Regression update: ALL valid neighbours (real road distance) ──────────
		for nb in G.get(node, []):
			edge = f"{node},{nb}"
			if edge not in Dist or edge not in Cost:
				continue
			n_data, sx, sy, sxx, sxy, syy, a, b = _online_reg_update(
				n_data, sx, sy, sxx, sxy, syy, float(Dist[edge]), float(Cost[edge])
			)

		# ── History record (after regression, before expansion) ──────────────────
		h_cur     = h.get(node, 0.0)
		rem_cur   = budget - e
		after_cur = rem_cur - (a * h_cur + b)
		if step >= WARMUP_STATES and after_cur > 0:
			h_para_rep = min(rem_cur / after_cur, float(MAX_PENALTY))
		elif step >= WARMUP_STATES:
			h_para_rep = float(MAX_PENALTY)
		else:
			h_para_rep = 1.0
		history.append((step, a, b, h_para_rep, rem_cur))

		# ── Search expansion ───────────────────────────────────────────────────
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
			best[nb] = [(ld, le) for ld, le in nb_labels if not (nd <= ld and ne <= le)]
			best[nb].append((nd, ne))
			parent[(nb, ne)] = (node, e)

			h_nb  = h.get(nb, float("inf"))
			rem   = budget - ne
			after = rem - (a * h_nb + b)
			if step >= WARMUP_STATES and after > 0:
				h_para = min(rem / after, float(MAX_PENALTY))
			else:
				h_para = 1.0
			heapq.heappush(pq, (nd + h_nb * h_para, nd, ne, nb))

	corr = _pearson_from_sums(n_data, sx, sy, sxx, sxy, syy)
	return float("inf"), -1, [], len(closed), (a, b, corr), history


def astar_constrained_pythagorean_energyaware_realdist(G, Dist, Cost, Coord, start, goal, budget, h):
	"""
	A* + Pythagorean heuristic + online energy-aware scaling (real-dist variant).

	Identical to the Haversine real-dist variant; h is the Pythagorean dict.
	Regression uses actual road distances from Dist.json for both 3e and 3f.
	Warmup and history tracking are the same.
	"""
	n_data = 0
	sx = sy = sxx = sxy = syy = 0.0
	a, b = 0.0, 0.0
	history = []

	pq = [(h.get(start, float("inf")), 0.0, 0, start)]
	best   = {start: [(0.0, 0)]}
	parent = {(start, 0): None}
	closed = set()

	while pq:
		f, d, e, node = heapq.heappop(pq)

		if (node, e) in closed:
			continue
		closed.add((node, e))
		step = len(closed)

		if node == goal:
			path, state = [], (goal, e)
			while state is not None:
				path.append(state[0])
				state = parent[state]
			h_cur     = h.get(node, 0.0)
			rem_cur   = budget - e
			after_cur = rem_cur - (a * h_cur + b)
			if step >= WARMUP_STATES and after_cur > 0:
				h_para_rep = min(rem_cur / after_cur, float(MAX_PENALTY))
			elif step >= WARMUP_STATES:
				h_para_rep = float(MAX_PENALTY)
			else:
				h_para_rep = 1.0
			history.append((step, a, b, h_para_rep, rem_cur))
			corr = _pearson_from_sums(n_data, sx, sy, sxx, sxy, syy)
			return d, e, path[::-1], len(closed), (a, b, corr), history

		# ── Regression update: ALL valid neighbours (real road distance) ──────────
		for nb in G.get(node, []):
			edge = f"{node},{nb}"
			if edge not in Dist or edge not in Cost:
				continue
			n_data, sx, sy, sxx, sxy, syy, a, b = _online_reg_update(
				n_data, sx, sy, sxx, sxy, syy, float(Dist[edge]), float(Cost[edge])
			)

		# ── History record (after regression, before expansion) ──────────────────
		h_cur     = h.get(node, 0.0)
		rem_cur   = budget - e
		after_cur = rem_cur - (a * h_cur + b)
		if step >= WARMUP_STATES and after_cur > 0:
			h_para_rep = min(rem_cur / after_cur, float(MAX_PENALTY))
		elif step >= WARMUP_STATES:
			h_para_rep = float(MAX_PENALTY)
		else:
			h_para_rep = 1.0
		history.append((step, a, b, h_para_rep, rem_cur))

		# ── Search expansion ───────────────────────────────────────────────────
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
			best[nb] = [(ld, le) for ld, le in nb_labels if not (nd <= ld and ne <= le)]
			best[nb].append((nd, ne))
			parent[(nb, ne)] = (node, e)

			h_nb  = h.get(nb, float("inf"))
			rem   = budget - ne
			after = rem - (a * h_nb + b)
			if step >= WARMUP_STATES and after > 0:
				h_para = min(rem / after, float(MAX_PENALTY))
			else:
				h_para = 1.0
			heapq.heappush(pq, (nd + h_nb * h_para, nd, ne, nb))

	corr = _pearson_from_sums(n_data, sx, sy, sxx, sxy, syy)
	return float("inf"), -1, [], len(closed), (a, b, corr), history


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
	t3c_dist, t3c_energy, t3c_path, t3c_states, t3c_lin, t3c_hist = astar_constrained_haversine_energyaware(
		G, Dist, Cost, Coord, START, GOAL, ENERGY_BUDGET, h_haversine
	)
	a_hav, b_hav, corr_hav = t3c_lin
	_plot_ea_history(t3c_hist, "Task 3c – Haversine + EA all-neighbours (online)", base_dir / "task3c_history.png")

	print()
	print("=" * 60)
	print("Task 3c: A* — Haversine + energy-aware (all-neighbours online)")
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

	# ── Task 3d: A* with Pythagorean + energy-aware (all-neighbours) ─────────
	t3d_dist, t3d_energy, t3d_path, t3d_states, t3d_lin, t3d_hist = astar_constrained_pythagorean_energyaware(
		G, Dist, Cost, Coord, START, GOAL, ENERGY_BUDGET, h_pythagorean
	)
	a_pyth, b_pyth, corr_pyth = t3d_lin
	_plot_ea_history(t3d_hist, "Task 3d – Pythagorean + EA all-neighbours (online)", base_dir / "task3d_history.png")

	print()
	print("=" * 60)
	print("Task 3d: A* — Pythagorean + energy-aware (all-neighbours online)")
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

	# ── Task 3e: A* with Haversine heuristic + energy-aware (real-dist) ────────
	t3e_dist, t3e_energy, t3e_path, t3e_states, t3e_lin, t3e_hist = astar_constrained_haversine_energyaware_realdist(
		G, Dist, Cost, Coord, START, GOAL, ENERGY_BUDGET, h_haversine
	)
	a_hav_p, b_hav_p, corr_hav_p = t3e_lin
	_plot_ea_history(t3e_hist, "Task 3e – Haversine heuristic + EA real-dist regression (online)", base_dir / "task3e_history.png")

	print()
	print("=" * 60)
	print("Task 3e: A* — Haversine heuristic + energy-aware (real-dist regression)")
	print(f"  Linearity (real road dist): cost ≈ {a_hav_p:.6f} * road_dist + {b_hav_p:.2f}")
	print(f"  Pearson correlation: {corr_hav_p:.6f}")
	if t3e_path:
		print(f"Shortest path: {'->'.join(t3e_path)}.")
		print(f"Shortest distance: {t3e_dist}.")
		print(f"Total energy cost: {t3e_energy}.")
		print(f"Number of nodes in path: {len(t3e_path)}.")
		print(f"Number of states visited: {t3e_states}.")
	else:
		print("No feasible path found within the energy budget.")

	# ── Task 3f: A* with Pythagorean heuristic + energy-aware (real-dist) ──────
	t3f_dist, t3f_energy, t3f_path, t3f_states, t3f_lin, t3f_hist = astar_constrained_pythagorean_energyaware_realdist(
		G, Dist, Cost, Coord, START, GOAL, ENERGY_BUDGET, h_pythagorean
	)
	a_pyth_p, b_pyth_p, corr_pyth_p = t3f_lin
	_plot_ea_history(t3f_hist, "Task 3f – Pythagorean heuristic + EA real-dist regression (online)", base_dir / "task3f_history.png")

	print()
	print("=" * 60)
	print("Task 3f: A* — Pythagorean heuristic + energy-aware (real-dist regression)")
	print(f"  Linearity (real road dist): cost ≈ {a_pyth_p:.6f} * road_dist + {b_pyth_p:.2f}")
	print(f"  Pearson correlation: {corr_pyth_p:.6f}")
	if t3f_path:
		print(f"Shortest path: {'->'.join(t3f_path)}.")
		print(f"Shortest distance: {t3f_dist}.")
		print(f"Total energy cost: {t3f_energy}.")
		print(f"Number of nodes in path: {len(t3f_path)}.")
		print(f"Number of states visited: {t3f_states}.")
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
		p("Task 3c: A* — Haversine + energy-aware (all-neighbours online)")
		p(f"Shortest path: {'->'.join(t3c_path) if t3c_path else 'No feasible path found within the energy budget'}.")
		p()
		p("Task 3d: A* — Pythagorean + energy-aware (all-neighbours online)")
		p(f"Shortest path: {'->'.join(t3d_path) if t3d_path else 'No feasible path found within the energy budget'}.")
		p()
		p("Task 3e: A* — Haversine heuristic + energy-aware (real-dist regression)")
		p(f"Shortest path: {'->'.join(t3e_path) if t3e_path else 'No feasible path found within the energy budget'}.")
		p()
		p("Task 3f: A* — Pythagorean heuristic + energy-aware (real-dist regression)")
		p(f"Shortest path: {'->'.join(t3f_path) if t3f_path else 'No feasible path found within the energy budget'}.")

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
		w("## Task 3c: A* — Haversine + energy-aware, all-neighbours online")
		w()
		w("Collects `(h[nb], edge_cost)` for **all** valid neighbours at each expansion.")
		w()
		w(f"- Final regression: cost ≈ {a_hav:.6f} × haversine_dist + {b_hav:.2f}")
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
		w("## Task 3d: A* — Pythagorean + energy-aware, all-neighbours online")
		w()
		w("Collects `(h[nb], edge_cost)` for **all** valid neighbours at each expansion.")
		w()
		w(f"- Final regression: cost ≈ {a_pyth:.6f} × pythagorean_dist + {b_pyth:.2f}")
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
		w("## Task 3e: A* — Haversine heuristic + energy-aware, real-dist regression")
		w()
		w("Collects `(real_road_dist, edge_cost)` for **all** valid neighbours; regression fits `cost ≈ a·road_dist + b`.")
		w()
		w(f"- Final regression: cost ≈ {a_hav_p:.6f} × haversine_dist + {b_hav_p:.2f}")
		w(f"- Pearson correlation: {corr_hav_p:.6f}")
		w()
		if t3e_path:
			w(f"- Shortest distance: {t3e_dist:.5f} m")
			w(f"- Total energy cost: {t3e_energy}")
			w(f"- Number of nodes in path: {len(t3e_path)}")
			w(f"- Number of states visited: {t3e_states}")
		else:
			w("No feasible path found within the energy budget.")

		w()
		w("## Task 3f: A* — Pythagorean heuristic + energy-aware, real-dist regression")
		w()
		w("Collects `(real_road_dist, edge_cost)` for **all** valid neighbours; regression fits `cost ≈ a·road_dist + b`.")
		w()
		w(f"- Final regression: cost ≈ {a_pyth_p:.6f} × pythagorean_dist + {b_pyth_p:.2f}")
		w(f"- Pearson correlation: {corr_pyth_p:.6f}")
		w()
		if t3f_path:
			w(f"- Shortest distance: {t3f_dist:.5f} m")
			w(f"- Total energy cost: {t3f_energy}")
			w(f"- Number of nodes in path: {len(t3f_path)}")
			w(f"- Number of states visited: {t3f_states}")
		else:
			w("No feasible path found within the energy budget.")

		w()
		w("## Comparison: states visited and path accuracy")
		w()
		hav_reduction     = (t2_states - t3a_states) / t2_states * 100
		pyth_reduction    = (t2_states - t3b_states) / t2_states * 100
		hav_ea_reduction  = (t2_states - t3c_states) / t2_states * 100
		pyth_ea_reduction = (t2_states - t3d_states) / t2_states * 100
		hav_p_reduction   = (t2_states - t3e_states) / t2_states * 100
		pyth_p_reduction  = (t2_states - t3f_states) / t2_states * 100
		t3a_accuracy  = t2_dist / t3a_dist * 100 if t3a_dist else 0.0
		t3b_accuracy  = t2_dist / t3b_dist * 100 if t3b_dist else 0.0
		t3c_accuracy  = t2_dist / t3c_dist * 100 if t3c_dist else 0.0
		t3d_accuracy  = t2_dist / t3d_dist * 100 if t3d_dist else 0.0
		t3e_accuracy  = t2_dist / t3e_dist * 100 if t3e_dist else 0.0
		t3f_accuracy  = t2_dist / t3f_dist * 100 if t3f_dist else 0.0
		w(f"| Algorithm                                      | States visited | Reduction vs UCS | Path optimality    |")
		w(f"|------------------------------------------------|----------------|------------------|--------------------|")
		w(f"| Task 2  UCS constrained (optimal)              | {t2_states:>14} | --               | 100.00% (baseline) |")
		w(f"| Task 3a A* Haversine                           | {t3a_states:>14} | {hav_reduction:>7.1f}%          | {t3a_accuracy:.2f}%             |")
		w(f"| Task 3b A* Pythagorean                         | {t3b_states:>14} | {pyth_reduction:>7.1f}%          | {t3b_accuracy:.2f}%             |")
		w(f"| Task 3c A* Haversine + EA all-nb online        | {t3c_states:>14} | {hav_ea_reduction:>7.1f}%          | {t3c_accuracy:.2f}%             |")
		w(f"| Task 3d A* Pythagorean + EA all-nb online      | {t3d_states:>14} | {pyth_ea_reduction:>7.1f}%          | {t3d_accuracy:.2f}%             |")
		w(f"| Task 3e A* Haversine heuristic + EA real-dist   | {t3e_states:>14} | {hav_p_reduction:>7.1f}%          | {t3e_accuracy:.2f}%             |")
		w(f"| Task 3f A* Pythagorean heuristic + EA real-dist | {t3f_states:>14} | {pyth_p_reduction:>7.1f}%          | {t3f_accuracy:.2f}%             |")


if __name__ == "__main__":
	main()
