# SC3000 Artificial Intelligence Lab 

NTU 2026 Sem 2.

---

## Informortion
- Contributor: 
  - Hung: Lab 1 Task 1.1, 1.2, 1.3 Haversine heuristic
  - Allen: Lab 1 Task 1.3 Pythagorean Heurstic
---

## Lab 1 Task 2.2 — Monte Carlo Q-Learning
- Author: Allen

### Standard Method

The standard definition of the Q-value is:

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[G_t \mid S_t = s,\ A_t = a\right]$$

My initial proposed approach was: in each episode, calculate 10 paths from start to end (ε-soft) and compute the average for each node. However, after researching the standard solution, the more common approach is to calculate Q-values using accumulated past paths (Sutton & Barto, 2018, pp. 100–101):

$$Q^{\pi_k}(s,a) = \frac{1}{k}\sum_{i=1}^{k}G_i$$

where $G_i$ is the return generated using policy $\pi_i$.

This does not strictly follow the definition above, since the mathematical definition specifies that Q should only be averaged over paths generated from the *current* policy, rather than accumulating past paths. However, it remains justifiable — each episode introduces only small policy improvements, so the cumulation still converges as the incremental changes are small enough.

This solution is stored in [`./Lab1/part2/task2-2`](./Lab1/part2/task2-2), reaching **90.9% similarity** with the optimal solution.

---

### Issues with the Standard Method

The standard cumulative method introduces two problems:

1. **Memory**: as episodes increase, storage grows linearly — $O(n)$.
2. **Staleness**: although policy changes are small per episode, accumulating all past returns can introduce errors over many episodes.

---

### Improved Method — Sliding-Window Returns (Task 2.2-v2)

To address both issues, I further improved the solution to average only over the **most recent 1000 returns**:

$$
Q^{\pi_k} = \begin{cases}
\dfrac{1}{k}\sum_{i=1}^{k}G_i & \text{if } k \leq 1000 \\
\dfrac{1}{1000}\sum_{i=k-1000}^{k}G_i & \text{if } k > 1000
\end{cases}
$$

A similar algorithmic idea can be found in other agent-related papers (Mnih et al., 2013).

This solution is stored in [`./Lab1/part2/task2-2-v2`](./Lab1/part2/task2-2-v2). The algorithm also reaches **90.9% similarity** with the optimal solution, but computed in a much shorter time — the time to average and space to store the data becomes $O(1)$.

---

### References

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press. http://incompleteideas.net/book/the-book-2nd.html

Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing Atari with deep reinforcement learning. *arXiv*. https://arxiv.org/abs/1312.5602
