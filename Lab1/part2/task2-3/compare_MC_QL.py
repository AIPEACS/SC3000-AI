"""
compare_MC_QL.py
================
Runs Monte Carlo v2 (sliding-window) and Q-Learning 100 times each
(20,000 episodes per run) and measures policy accuracy against the
VI-optimal policy from Task 2-1.

Outputs (both in task2-3/):
  compare_MC_QL_history.png  — per-run accuracy line chart for both algorithms
  compare_MC_QL.md           — mean / variance / per-run table
"""

import sys
import os
import json
import time
import contextlib
import importlib.util

import numpy as np
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))   # task2-3/
_PART2    = os.path.dirname(_HERE)                        # part2/
_TASK22   = os.path.join(_PART2, 'task2-2')
_TASK22V2 = os.path.join(_PART2, 'task2-2-v2')

for p in (_PART2, _TASK22, _TASK22V2, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── module loader (suppresses at-import prints & stdout redirects) ─────────────
def _load(mod_name, filepath):
    """Load a module by file path, suppressing any output it emits at import time."""
    spec = importlib.util.spec_from_file_location(mod_name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    with open(os.devnull, 'w', encoding='utf-8') as _dev, contextlib.redirect_stdout(_dev):
        spec.loader.exec_module(mod)
    return mod


_mc  = _load('_mc_v2_main', os.path.join(_TASK22V2, 'main.py'))
_ql  = _load('_ql_main',    os.path.join(_HERE,      'main.py'))

import scene_map as map0  # noqa: E402 (loaded after path setup)


# ── load VI optimal policy ────────────────────────────────────────────────────
_OPT_PATH = os.path.join(_PART2, 'task2-1', 'visualization',
                         'task1_Optimal_action_tensor.json')
with open(_OPT_PATH, encoding='utf-8') as _f:
    _opt = json.load(_f)
_t = _opt['action_tensor']
OPTIMAL = {(x, y): int(_t[x][y]) for x in range(5) for y in range(5)}

VALID_STATES = [
    (x, y) for x in range(5) for y in range(5)
    if (x, y) not in map0.road_blocking and (x, y) != map0.end_point
]


def accuracy(policy):
    """Return % of valid states where policy matches the VI optimal."""
    return 100.0 * sum(1 for s in VALID_STATES if policy[s] == OPTIMAL[s]) / len(VALID_STATES)


# ── experiment ────────────────────────────────────────────────────────────────
NUM_RUNS     = 50
NUM_EPISODES = 20_000

mc_acc = []
ql_acc = []

print(f"\nComparing MC v2 vs Q-Learning — {NUM_RUNS} runs × {NUM_EPISODES:,} episodes each")
print(f"Valid states evaluated: {len(VALID_STATES)}  |  ε = 0.1\n")

t0 = time.time()
with open(os.devnull, 'w', encoding='utf-8') as _dev:
    for run in range(NUM_RUNS):

        # ── Monte Carlo v2 ──
        with contextlib.redirect_stdout(_dev):
            _, mc_policy, _, _ = _mc.monte_carlo_control(
                num_episodes=NUM_EPISODES, epsilon=0.1
            )
        mc_acc.append(accuracy(mc_policy))

        # ── Q-Learning ──
        with contextlib.redirect_stdout(_dev):
            Q_ql, _ = _ql.q_learning(NUM_EPISODES, _ql.EPSILON, _ql.ALPHA)
        ql_policy = _ql.extract_policy_from_q(Q_ql)
        ql_acc.append(accuracy(ql_policy))

        elapsed = time.time() - t0
        print(
            f"  [{run + 1:3d}/{NUM_RUNS}]  "
            f"MC={mc_acc[-1]:.1f}%  QL={ql_acc[-1]:.1f}%  "
            f"elapsed={elapsed:.0f}s",
            flush=True
        )

mc_arr = np.array(mc_acc)
ql_arr = np.array(ql_acc)

# ── history plot ──────────────────────────────────────────────────────────────
runs = list(range(1, NUM_RUNS + 1))

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(runs, mc_arr, color='steelblue',  alpha=0.75, linewidth=1.2,
        marker='o', markersize=3, label='Monte Carlo v2 (sliding window)')
ax.plot(runs, ql_arr, color='darkorange', alpha=0.75, linewidth=1.2,
        marker='s', markersize=3, label='Q-Learning')
ax.axhline(mc_arr.mean(), color='steelblue',  linestyle='--', linewidth=1.8,
           alpha=0.9, label=f'MC mean = {mc_arr.mean():.1f}%')
ax.axhline(ql_arr.mean(), color='darkorange', linestyle='--', linewidth=1.8,
           alpha=0.9, label=f'QL mean = {ql_arr.mean():.1f}%')

ax.set_xlabel('Run', fontsize=12)
ax.set_ylabel('Accuracy vs Optimal Policy (%)', fontsize=12)
ax.set_title(
    f'MC v2 vs Q-Learning — Policy Accuracy over {NUM_RUNS} Independent Runs\n'
    f'({NUM_EPISODES:,} episodes/run, ε = 0.1)',
    fontsize=13, fontweight='bold'
)
ax.set_ylim([0, 105])
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()

png_path = os.path.join(_HERE, 'compare_MC_QL_history.png')
fig.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n✓  History plot  → {png_path}")

# ── markdown ──────────────────────────────────────────────────────────────────
md_path = os.path.join(_HERE, 'compare_MC_QL.md')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write("# Monte Carlo v2 vs Q-Learning: Statistical Comparison\n\n")
    f.write(f"- **Runs**: {NUM_RUNS}\n")
    f.write(f"- **Episodes per run**: {NUM_EPISODES:,}\n")
    f.write(f"- **Exploration rate (ε)**: 0.1\n")
    f.write(f"- **Valid states evaluated**: {len(VALID_STATES)} (excludes obstacles and goal)\n\n")

    f.write("## Summary Statistics\n\n")
    f.write("| Metric | Monte Carlo v2 | Q-Learning |\n")
    f.write("|--------|:--------------:|:----------:|\n")
    f.write(f"| Mean accuracy       | {mc_arr.mean():.2f}%  | {ql_arr.mean():.2f}%  |\n")
    f.write(f"| Standard deviation  | {mc_arr.std():.2f}%   | {ql_arr.std():.2f}%   |\n")
    f.write(f"| Variance            | {mc_arr.var():.4f}    | {ql_arr.var():.4f}    |\n")
    f.write(f"| Min                 | {mc_arr.min():.2f}%   | {ql_arr.min():.2f}%   |\n")
    f.write(f"| Max                 | {mc_arr.max():.2f}%   | {ql_arr.max():.2f}%   |\n")
    f.write(f"| Median              | {np.median(mc_arr):.2f}% | {np.median(ql_arr):.2f}% |\n\n")

    f.write("## Per-Run Accuracy\n\n")
    f.write("| Run | MC v2 (%) | Q-Learning (%) |\n")
    f.write("|:---:|:---------:|:--------------:|\n")
    for i, (mc, ql) in enumerate(zip(mc_acc, ql_acc), 1):
        f.write(f"| {i:3d} | {mc:.1f} | {ql:.1f} |\n")

print(f"✓  Markdown       → {md_path}")
