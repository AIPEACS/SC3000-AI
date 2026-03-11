# Lab1 descrption

# part 1

## A* Heuristic — Haversine Formula

A* requires an **admissible heuristic** h(n): an estimate of the remaining distance from node n to the goal that **never overestimates** the true cost.

Since the graph represents real-world locations with GPS coordinates, the heuristic used is the **Haversine formula** — the great-circle (straight-line) distance between two points on the Earth's surface.

### Why Haversine?
- Road distances are always ≥ straight-line distances, so Haversine never overestimates → **admissible**.
- It accounts for the Earth's curvature, making it more accurate than simple Euclidean distance for geographic coordinates.

### Formula

Given two points with latitude/longitude $(φ_1, λ_1)$ and $(φ_2, λ_2)$ in radians:

$$a = \sin^2\!\left(\frac{φ_2 - φ_1}{2}\right) + \cos φ_1 \cdot \cos φ_2 \cdot \sin^2\!\left(\frac{λ_2 - λ_1}{2}\right)$$

$$d = 2R \cdot \arcsin(\sqrt{a})$$

where $R = 6{,}371{,}000$ m (Earth's mean radius).

### Implementation detail
Coordinates in `Coord.json` are stored as integers = degrees × 10⁶, so they are divided by `1e6` before conversion to radians:

```python
R = 6371000.0
lon1, lat1 = vals[0] / 1e6, vals[1] / 1e6
dlat = lat2_r - lat1_r
dlon = lon2_r - lon1_r
a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
h[node] = 2 * R * math.asin(math.sqrt(a))
```

The heuristic is precomputed once for all nodes → goal before the search begins, giving O(1) lookup during A*.

# part 2