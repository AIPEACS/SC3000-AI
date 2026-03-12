import json, math

def t_cdf(t_val, df):
    """Compute two-tailed t-distribution CDF using incomplete beta function approximation."""
    # For large df, t-distribution approaches normal, but we use the exact formula
    # via incomplete beta: P(T <= t) based on beta regularization
    x = df / (df + t_val**2)
    # Incomplete beta I_x(a, b) with a = df/2, b = 0.5
    # For computational purposes when scipy unavailable:
    # Use the fact that for large df and |t| >> 1, p -> 0 exponentially
    # But we can use a better approximation from Abramowitz & Stegun
    
    # Satterthwaite approximation or normal approx works for large df
    # For df=730k, t distribution ~= normal distribution
    # So use survival function: P(T > |t|) ≈ P(Z > |t|) from normal
    
    t_abs = abs(t_val)
    if df > 30:  # Normal approximation is excellent
        # Standard normal approximation using error function
        import math
        # erf-based normal CDF: P(Z <= z) = 0.5 * (1 + erf(z/sqrt(2)))
        # In Python 3.2+: math.erf is available
        z_cdf = 0.5 * (1 + math.erf(t_abs / math.sqrt(2)))
        p_two_tailed = 2 * (1 - z_cdf)
        return p_two_tailed
    else:
        # Fall back to approximation for small df
        return None

with open('Coord.json') as f: Coord = json.load(f)
with open('Cost.json') as f: Cost = json.load(f)

xs, ys = [], []
for edge, cost in Cost.items():
    n1, n2 = edge.split(',')
    if n1 not in Coord or n2 not in Coord: continue
    lon1, lat1 = Coord[n1][0]/1e6, Coord[n1][1]/1e6
    lon2, lat2 = Coord[n2][0]/1e6, Coord[n2][1]/1e6
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    dist = math.sqrt(dlat**2 + dlon**2)
    xs.append(dist)
    ys.append(float(cost))

n = len(xs)
mx, my = sum(xs)/n, sum(ys)/n
Sxx = sum((x-mx)**2 for x in xs)
Syy = sum((y-my)**2 for y in ys)
Sxy = sum((x-mx)*(y-my) for x,y in zip(xs,ys))

r = Sxy / math.sqrt(Sxx*Syy)
t = r * math.sqrt(n-2) / math.sqrt(1 - r**2)
df = n - 2

# Two-tailed p-value
p = t_cdf(t, df)
t_abs = abs(t)

# Also compute regression slope for completeness
a = Sxy / Sxx
b = my - a * mx

print(f'Sample size: {n}')
print(f'Pearson r: {r:.6f}')
print(f'R-squared: {r**2:.6f}')
print(f't-statistic: {t:.4f}')
print(f'Degrees of freedom: {df}')
print(f'p-value (two-tailed): {p}')
if p == 0.0:
    # p-value underflowed; estimate lower bound
    # For standard normal: P(Z>t) ≈ min(1/sqrt(2π) * exp(-t²/2), float precision limit)
    import math
    log_p = -t_abs**2 / 2 - 0.5 * math.log(2 * math.pi)
    print(f'  (underflowed; log(p) ≈ {log_p:.2f}, p < 1e-300)')
print(f'\nRegression: Cost = {a:.6f} * distance + {b:.6f}')
