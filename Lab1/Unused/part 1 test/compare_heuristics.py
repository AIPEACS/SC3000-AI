"""
For every directed edge in Dist.json, prints:
  - stored Dist value
  - Haversine heuristic (straight-line, decimetres)
  - Pythagorean heuristic (flat-earth, decimetres)
  - which heuristic is larger
  - whether each heuristic is admissible (heuristic <= stored dist)

Uses Coord.json and Dist.json from the original part1 folder.
"""

import json
import math
from pathlib import Path

R_DM = 63_710_000.0   # Earth radius in decimetres

base = Path(__file__).resolve().parent.parent / "part1"

with open(base / "Coord.json") as f:
    Coord = json.load(f)
with open(base / "Dist.json") as f:
    Dist = json.load(f)


def haversine(u, v):
    lon1, lat1 = Coord[u][0] / 1e6, Coord[u][1] / 1e6
    lon2, lat2 = Coord[v][0] / 1e6, Coord[v][1] / 1e6
    φ1, λ1 = math.radians(lat1), math.radians(lon1)
    φ2, λ2 = math.radians(lat2), math.radians(lon2)
    a = math.sin((φ2-φ1)/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin((λ2-λ1)/2)**2
    return 2 * R_DM * math.asin(math.sqrt(a))


def pythagorean(u, v):
    lon1, lat1 = Coord[u][0] / 1e6, Coord[u][1] / 1e6
    lon2, lat2 = Coord[v][0] / 1e6, Coord[v][1] / 1e6
    return math.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 1_000_000


header = (f"{'Edge':<12} {'Stored Dist':>13} {'Haversine':>13} "
          f"{'Pythagorean':>13} {'Larger':>11} {'Hav Admis?':>11} {'Pyth Admis?':>12}")
print(header)
print("-" * len(header))

hav_violations = 0
pyth_violations = 0

for edge, stored_val in Dist.items():
    u, v = edge.split(",")
    if u not in Coord or v not in Coord:
        continue
    stored = float(stored_val)
    hav   = haversine(u, v)
    pyth  = pythagorean(u, v)

    larger      = "Haversine"  if hav  >  pyth else ("Pythagorean" if pyth > hav else "Equal")
    hav_admis   = hav  <= stored
    pyth_admis  = pyth <= stored
    if not hav_admis:  hav_violations  += 1
    if not pyth_admis: pyth_violations += 1

    print(f"{edge:<12} {stored:>13.2f} {hav:>13.2f} {pyth:>13.2f} "
          f"{larger:>11} {'Yes' if hav_admis else 'NO':>11} {'Yes' if pyth_admis else 'NO':>12}")

total = len(Dist)
print()
print(f"Total edges: {total}")
print(f"Haversine   admissibility violations: {hav_violations}  ({hav_violations/total*100:.1f}%)")
print(f"Pythagorean admissibility violations: {pyth_violations}  ({pyth_violations/total*100:.1f}%)")
