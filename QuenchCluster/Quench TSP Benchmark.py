"""
Quench-Cluster TSP Benchmark
Author: Mohamed Noureldin
Verification Script v2.0
Run in Google Colab: pip install numpy scipy scikit-learn matplotlib pandas
"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# ── Held-Karp Lower Bound (for small N only, as reference) ──────────────────
def held_karp(dist_matrix):
    """Exact TSP via dynamic programming. Only feasible for N <= 20."""
    n = len(dist_matrix)
    C = {}
    for k in range(1, n):
        C[(1 << k, k)] = (dist_matrix[0][k], 0)
    for subset_size in range(2, n):
        for subset in combinations(range(1, n), subset_size):
            bits = sum(1 << bit for bit in subset)
            for k in subset:
                prev = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == k: continue
                    if (prev, m) in C:
                        res.append((C[(prev, m)][0] + dist_matrix[m][k], m))
                if res:
                    C[(bits, k)] = min(res)
    bits = (2**n - 1) & ~1
    res = [(C[(bits, k)][0] + dist_matrix[k][0], k) for k in range(1, n) if (bits, k) in C]
    return min(res)[0] if res else float('inf')

# ── Greedy Nearest Neighbor (Baseline) ──────────────────────────────────────
def greedy_nn(points):
    N = len(points)
    dist = cdist(points, points)
    visited = np.zeros(N, dtype=bool)
    tour = [0]
    visited[0] = True
    total = 0.0
    for _ in range(N - 1):
        last = tour[-1]
        dist[last, visited] = np.inf
        nxt = np.argmin(dist[last])
        total += dist[last, nxt]
        tour.append(nxt)
        visited[nxt] = True
    total += cdist(points[tour[-1]:tour[-1]+1], points[tour[0]:tour[0]+1])[0, 0]
    return np.array(tour), total

# ── 2-opt Local Search ───────────────────────────────────────────────────────
def two_opt(points, tour, max_iter=500):
    best = tour.copy()
    best_dist = tour_length(points, best)
    improved = True
    iterations = 0
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                new_tour = np.concatenate([best[:i], best[i:j+1][::-1], best[j+1:]])
                new_dist = tour_length(points, new_tour)
                if new_dist < best_dist - 1e-10:
                    best, best_dist = new_tour, new_dist
                    improved = True
    return best, best_dist

def tour_length(points, tour):
    t = np.append(tour, tour[0])
    return np.sum(np.linalg.norm(points[t[:-1]] - points[t[1:]], axis=1))

# ── Quench-Cluster Core ──────────────────────────────────────────────────────
def quench_cluster_tsp(points, k=None, refine=True):
    N = len(points)
    if k is None:
        k = max(4, int(np.sqrt(N)))

    # Phase 1: Quench (Parallel K-Means nucleation)
    km = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(points)
    labels = km.labels_

    # Phase 2: Local Solidification (solve each crystal)
    cluster_tours = {}
    cluster_costs = {}
    for cid in range(k):
        mask = np.where(labels == cid)[0]
        if len(mask) == 0:
            continue
        cp = points[mask]
        if len(mask) == 1:
            cluster_tours[cid] = mask
            cluster_costs[cid] = 0.0
        else:
            local_tour, local_cost = greedy_nn(cp)
            if refine and len(mask) <= 200:
                local_tour, local_cost = two_opt(cp, local_tour, max_iter=100)
            cluster_tours[cid] = mask[local_tour]
            cluster_costs[cid] = local_cost

    # Phase 3: Meta-TSP on centroids (Pyramid merge)
    valid_cids = list(cluster_tours.keys())
    centroids = np.array([points[cluster_tours[cid]].mean(axis=0) for cid in valid_cids])
    meta_tour, _ = greedy_nn(centroids)
    ordered_cids = [valid_cids[i] for i in meta_tour]

    # Phase 4: Intelligent stitching with rotation
    global_tour = []
    for cid in ordered_cids:
        ctour = cluster_tours[cid]
        if len(global_tour) > 0:
            last_pt = global_tour[-1]
            dists = np.linalg.norm(points[ctour] - points[last_pt], axis=1)
            best_start = np.argmin(dists)
            ctour = np.roll(ctour, -best_start)
        global_tour.extend(ctour.tolist())

    global_tour = np.array(global_tour)
    dist = tour_length(points, global_tour)
    return global_tour, dist

# ── Full Benchmark ───────────────────────────────────────────────────────────
def run_full_benchmark(sizes=[500, 1000, 2000, 5000, 10000], seeds=3):
    results = []
    print(f"\n{'='*75}")
    print(f"{'N':>8} | {'Greedy(s)':>10} | {'Greedy Dist':>12} | {'Quench(s)':>10} | {'Quench Dist':>12} | {'Speedup':>8} | {'Gap%':>7}")
    print(f"{'='*75}")

    for N in sizes:
        g_times, g_dists = [], []
        q_times, q_dists = [], []

        for seed in range(seeds):
            np.random.seed(seed * 42)
            pts = np.random.rand(N, 2) * 1000

            # Baseline: Greedy NN
            t0 = time.perf_counter()
            _, gd = greedy_nn(pts)
            g_times.append(time.perf_counter() - t0)
            g_dists.append(gd)

            # Quench-Cluster
            t0 = time.perf_counter()
            _, qd = quench_cluster_tsp(pts)
            q_times.append(time.perf_counter() - t0)
            q_dists.append(qd)

        gt, gd = np.mean(g_times), np.mean(g_dists)
        qt, qd = np.mean(q_times), np.mean(q_dists)
        speedup = gt / qt
        gap = ((qd - gd) / gd) * 100  # negative = Quench is BETTER

        results.append({
            'N': N,
            'Greedy_Time': gt, 'Greedy_Dist': gd,
            'Quench_Time': qt, 'Quench_Dist': qd,
            'Speedup': speedup, 'Quality_Gap%': gap
        })
        print(f"{N:>8} | {gt:>10.4f} | {gd:>12.1f} | {qt:>10.4f} | {qd:>12.1f} | {speedup:>8.2f}x | {gap:>+7.2f}%")

    print(f"{'='*75}")
    print("Note: Quality Gap% < 0 means Quench produces SHORTER tour than Greedy.")
    return pd.DataFrame(results)

# ── Visualization ─────────────────────────────────────────────────────────────
def plot_results(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Quench-Cluster TSP vs Greedy Baseline", fontsize=14, fontweight='bold')

    axes[0].plot(df['N'], df['Greedy_Time'], 'r--o', label='Greedy NN (Baseline)')
    axes[0].plot(df['N'], df['Quench_Time'], 'g-o', label='Quench-Cluster')
    axes[0].set_xlabel('Number of Cities (N)')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].set_title('Execution Time Scaling')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df['N'], df['Speedup'], 'b-o')
    axes[1].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Baseline (1x)')
    axes[1].set_xlabel('Number of Cities (N)')
    axes[1].set_ylabel('Speedup (x times)')
    axes[1].set_title('Speedup Factor vs N')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df['N'], df['Quality_Gap%'], 'purple', marker='o')
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Number of Cities (N)')
    axes[2].set_ylabel('Quality Gap % (negative = Quench is better)')
    axes[2].set_title('Tour Quality vs Greedy')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quench_cluster_benchmark.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart saved: quench_cluster_benchmark.png")

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = run_full_benchmark(sizes=[500, 1000, 2000, 5000, 10000], seeds=3)
    plot_results(df)
    df.to_csv('quench_cluster_results.csv', index=False)
    print("\nResults saved to quench_cluster_results.csv")
