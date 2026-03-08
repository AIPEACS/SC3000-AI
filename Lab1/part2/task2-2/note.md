# Monte Carlo Control — Implementation Notes

## What the current code does (Standard MC Control, Sutton & Barto Ch.5)

- Runs **1 episode per iteration** under the current ε-greedy policy.
- For each first-visited (s, a) pair in the episode, the discounted return G_t is appended to `returns[(s,a)]`.
- Q(s,a) is updated as the **mean of all past returns** for that pair — including returns from all previous (different) policies.
- The policy implicitly improves each episode since it is always greedy w.r.t. the latest Q.

### Known theoretical issue

This mixes returns from different policies:
- G0 was sampled under π0, G1 under π1, G2 under π2, ...
- But Q2(s,a) = mean(G0, G1, G2), which violates Q^π(s,a) = E_π[G_t | s,a] strictly.

The standard algorithm accepts this as an approximation. It converges in practice because:
1. ε-greedy keeps policy changes small and gradual.
2. Near convergence the policy barely changes, so late returns dominate and correct the average.

This is a pragmatic algorithm, not theoretically pure.

## Alternative approach (suggested by student)

A cleaner conceptual approach would be:
1. Fix policy π_k.
2. Run N trajectories (e.g. 10) under π_k.
3. Average their returns → get Q^{π_k}(s,a) properly.
4. Improve policy greedily: π_{k+1} = greedy(Q^{π_k}).
5. Repeat from step 1, **discarding all old returns**.

This properly satisfies Q^π(s,a) = E_π[G_t | s,a] since all samples come from the same π.

The theoretically rigorous version for mixing samples across policies would require **off-policy MC with importance sampling** — significantly more complex and not required by this assignment.

## Current code status

The current implementation follows the **standard algorithm** as required by the assignment spec.
No changes needed.
