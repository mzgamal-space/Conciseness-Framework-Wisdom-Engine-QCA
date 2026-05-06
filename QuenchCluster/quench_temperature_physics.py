"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  QUENCH-CLUSTER — TEMPERATURE FUNCTIONS                                      ║
║  Replaces the empirically-tuned Noureldin formula with physics-derived       ║
║  alternatives. Original formula retained for benchmark comparison.           ║
║                                                                              ║
║  Mohamed Gamal Eldin Abdelaziz Noureldin — Conciseness Framework             ║
║  2026                                                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PROBLEM WITH THE ORIGINAL FORMULA                                           ║
║                                                                              ║
║  T_q_noureldin = (N / 24r) ^ (1/π²)                                          ║
║                                                                              ║
║  The constants 24 and exponent 1/π² ≈ 0.101 are empirically tuned and       ║
║  have no derivation from first-principles thermodynamics.                    ║
║                                                                              ║
║  Critical defect: the formula produces T_q ≈ 0.7 – 1.2 regardless of        ║
║  the coordinate scale of the problem. For Logistics nodes on [0,1000],       ║
║  T_q ≈ 0.71 while distances range up to ~1414. This yields                   ║
║  P_bind(d_mean) ≈ 10^-176 — essentially zero for all pairs.                  ║
║                                                                              ║
║  The algorithm only survives this because the downstream softmax             ║
║  normalisation recovers nearest-neighbour assignment from numerically        ║
║  degenerate Boltzmann weights. Correct physics should not require rescue     ║
║  by downstream normalisation.                                                ║
║                                                                              ║
║  RECOMMENDED REPLACEMENT: T_boltzmann() — derived from first principles.    ║
║  Original formula preserved as T_noureldin() for comparison.                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import math
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 0.  NODE TYPE (minimal — matches Quench-Cluster data structure)
# ─────────────────────────────────────────────────────────────────────────────

class _NodeLike:
    """Minimal interface used by temperature functions."""
    def __init__(self, coords: np.ndarray):
        self.coords = coords


def _mean_nn_distance(nodes: list, sample: int = 300) -> tuple:
    """
    Compute mean nearest-neighbour distance and distance standard deviation
    from a random sample of nodes.

    Returns
    -------
    d_nn   : float  — mean nearest-neighbour distance (natural length scale)
    sigma_D: float  — standard deviation of pairwise distances in sample
    """
    coords = np.array([n.coords[:2] for n in nodes], dtype=float)
    N = len(coords)
    if N <= 1:
        return 1.0, 1.0

    idx = np.random.choice(N, min(N, sample), replace=False)
    sub = coords[idx]
    diff = sub[:, None, :] - sub[None, :, :]
    dists = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)

    d_nn    = float(np.mean(dists.min(axis=1)))
    # sigma over all off-diagonal pairs
    finite  = dists[dists < np.inf]
    sigma_D = float(np.std(finite)) if len(finite) > 0 else 1.0

    return max(d_nn, 1e-9), max(sigma_D, 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  T_BOLTZMANN  (RECOMMENDED PRIMARY)
# ─────────────────────────────────────────────────────────────────────────────

def T_boltzmann(nodes: list, k: int) -> float:
    """
    Boltzmann Mean-Field Critical Temperature.

    Derivation
    ----------
    The binding probability in a Boltzmann system is:

        P_bind(d) = exp(-d / T)

    We want node i to bind preferentially to its cluster of k_cluster = N/K
    nearest neighbours and not to the rest. The critical temperature T* is
    defined as the temperature at which the probability of binding at the
    mean nearest-neighbour distance d_nn equals exactly 1/k_cluster:

        P_bind(d_nn) = 1/k_cluster
        exp(-d_nn / T*) = 1/k_cluster
        -d_nn / T* = -ln(k_cluster)
        T* = d_nn / ln(k_cluster)

    Physical interpretation
    -----------------------
    Below T*, each node binds with probability > 1/k_cluster to nodes within
    d_nn — cluster formation is favoured. Above T*, distant binding dominates
    and the system stays disordered. T* is the exact crystallisation threshold.

    Selectivity ratio
    -----------------
    P_bind(d_nn) / P_bind(2*d_nn) = exp(d_nn/T*) = k_cluster.
    A node is k_cluster times more likely to bind within its cluster than
    to a node at twice the nearest-neighbour distance. This is exactly the
    selectivity required for K balanced clusters.

    Scale invariance
    ----------------
    Both d_nn and T* scale with the coordinate units of the problem.
    If coordinates are doubled, T* doubles and all binding probabilities
    are unchanged — the formula is dimensionally consistent.

    Parameters
    ----------
    nodes : list of Node objects (must have .coords attribute)
    k     : target number of clusters

    Returns
    -------
    T_B : float — Boltzmann critical temperature in coordinate units
    """
    N = len(nodes)
    if N <= 1:
        return 1.0

    k_cluster = max(N / max(k, 1), 2.0)  # expected nodes per cluster
    d_nn, _ = _mean_nn_distance(nodes)

    T_B = d_nn / math.log(k_cluster)
    return max(T_B, 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  T_REM  (SECONDARY — more permissive / warmer start)
# ─────────────────────────────────────────────────────────────────────────────

def T_rem(nodes: list, k: int) -> float:
    """
    Random Energy Model (REM) Critical Temperature.

    Derivation
    ----------
    Derrida (1981) proved that for a system of N states whose energies are
    i.i.d. Gaussian with standard deviation σ_E, the system undergoes an
    abrupt freezing transition at:

        T_c = σ_E / sqrt(2 ln N)

    Below T_c the system is frozen in its ground-state cluster (crystal);
    above T_c it is in the liquid (disordered) phase.

    Adapted to the quench-cluster context:
    - 'N states' → k_cluster = N/K nodes competing for binding
    - 'energies' → distances D_ij (the cost landscape)
    - σ_E        → σ_D (standard deviation of pairwise distances)

        T_REM = σ_D / sqrt(2 ln k_cluster)

    When to prefer REM over Boltzmann
    ----------------------------------
    T_REM uses the full distance distribution σ_D rather than only the
    nearest-neighbour distance d_nn. It is appropriate when the distance
    distribution is wide (highly non-uniform point clouds) and you want
    a warmer start that allows exploration before crystallisation, analogous
    to a slower quench rate in materials science.

    Reference
    ---------
    Derrida, B. (1981). Random-energy model: An exactly solvable model of
    disordered systems. Physical Review B, 24(5), 2613.

    Parameters
    ----------
    nodes : list of Node objects
    k     : target number of clusters

    Returns
    -------
    T_REM : float — REM critical temperature in coordinate units
    """
    N = len(nodes)
    if N <= 1:
        return 1.0

    k_cluster = max(N / max(k, 1), 2.0)
    _, sigma_D = _mean_nn_distance(nodes)

    T_REM = sigma_D / math.sqrt(2.0 * math.log(k_cluster))
    return max(T_REM, 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  T_NOURELDIN  (ORIGINAL — preserved for benchmark comparison)
# ─────────────────────────────────────────────────────────────────────────────

def T_noureldin(n: int, r: float) -> float:
    """
    Original Noureldin Quench Temperature (empirically tuned).

    Formula
    -------
    T_q = (N / 24r^2) ^ (1/π²)

    Status
    ------
    This formula was introduced in Quench-Cluster v1 as a practical
    approximation. Its constants (24, 1/π²) have no derivation from
    first-principles thermodynamics. They were found empirically to produce
    acceptable clustering on TSP instances.

    Known limitations (documented in QCA_Corrected_v2.docx)
    --------------------------------------------------------
    1. Scale blindness: T_q ≈ 0.7 – 1.2 regardless of the coordinate scale
       of the problem. For Logistics on [0,1000], T_q ≈ 0.71 while distances
       reach ~1414, so P_bind(d_mean) ≈ 10^-176.

    2. Effective behaviour: the formula acts as a hard nearest-neighbour
       selector via downstream softmax normalisation, not as a true Boltzmann
       temperature. This works but obscures the physical mechanism.

    3. The exponent 1/π² makes T_q very insensitive to N. A 10× change
       in N/r produces only ~26% change in T_q.

    Retained for
    ------------
    - Benchmarking new temperature functions against original behaviour
    - Investigating whether the scale-insensitivity is incidentally beneficial
      in high-dimensional or non-Euclidean domains

    Parameters
    ----------
    n : int   — number of nodes
    r : float — search radius (typically mean std of coordinates × radius_scale)

    Returns
    -------
    T_q : float — Noureldin quench temperature
    """
    return (n / max(24.0 * r**2, 1e-9)) ** (1.0 / math.pi ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  UNIFIED SELECTOR  (drop-in replacement for _phi_mce internals)
# ─────────────────────────────────────────────────────────────────────────────

TEMP_METHODS = ("boltzmann", "rem", "noureldin")


def quench_temperature(
    nodes: list,
    k: int,
    radius_scale: float = 1.0,
    method: str = "boltzmann",
) -> float:
    """
    Unified temperature selector for HybridNucleation._phi_mce().

    Parameters
    ----------
    nodes        : list of Node objects with .coords attribute
    k            : target number of clusters
    radius_scale : scale multiplier used by HybridNucleation (only affects
                   noureldin method; the physics methods derive r from data)
    method       : one of "boltzmann" | "rem" | "noureldin"
                   Default: "boltzmann" (recommended)

    Returns
    -------
    T : float — quench temperature in the coordinate units of the problem

    Notes
    -----
    Swap method="boltzmann" ↔ method="rem" ↔ method="noureldin" in
    HybridQuenchCluster.__init__ or per-run in the benchmark to compare.

    Physical guarantee
    ------------------
    All three methods return T > 0.  Boltzmann and REM scale with the
    coordinate geometry of the problem and are therefore dimensionally
    consistent. Noureldin does not scale but is retained for comparison.
    """
    assert method in TEMP_METHODS, (
        f"method must be one of {TEMP_METHODS}, got '{method}'"
    )

    if method == "boltzmann":
        return T_boltzmann(nodes, k)

    elif method == "rem":
        return T_rem(nodes, k)

    else:  # "noureldin"
        coords = np.array([n.coords[:2] for n in nodes], dtype=float)
        N = len(coords)
        r = float(np.mean(np.std(coords, axis=0))) * radius_scale + 1e-9
        return T_noureldin(N, r)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  INTEGRATION PATCH for HybridNucleation._phi_mce  (v4 hybrid)
# ─────────────────────────────────────────────────────────────────────────────
#
# In quench_cluster_v4_hybrid.py, replace the existing _phi_mce temperature
# calculation:
#
#   BEFORE (lines inside _phi_mce):
#       r  = float(np.mean(np.std(coords, axis=0))) * radius_scale + 1e-9
#       T_q = (N / max(24.0 * r**2, 1e-9)) ** (1.0 / math.pi**2)
#
#   AFTER:
#       from quench_temperature_physics import quench_temperature
#       T_q = quench_temperature(nodes, k, radius_scale, method=self.temp_method)
#
# In HybridQuenchCluster.__init__, add:
#       self.temp_method = temp_method   # default "boltzmann"
#
# This makes the temperature method a per-engine configuration parameter,
# enabling direct A/B benchmark comparison in run_demo().
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 6.  BENCHMARK — three methods head-to-head
# ─────────────────────────────────────────────────────────────────────────────

class _MockNode:
    def __init__(self, coords):
        self.coords = coords


def _run_benchmark():
    """
    Compares T_boltzmann, T_rem, T_noureldin on four representative domains.
    Reports: temperature, binding probability at d_nn and 2*d_nn, selectivity.
    """
    configs = [
        ("Logistics  (N=250, [0,1000])",  250, 15, np.random.rand(250,2)*1000),
        ("Chip       (N=200, [0,100])",   200, 13, np.random.rand(200,2)*100),
        ("Protein    (N=60,  N(0,5))",     60,  7, np.random.randn(60,2)*5),
        ("Search     (N=180, N(0,1))",    180, 13, np.random.randn(180,2)),
    ]

    hdr = (
        f"\n{'Domain':35s} {'Method':12s}"
        f" {'T':>9} {'P(d_nn)':>10} {'P(2*d_nn)':>11} {'Select.':>10}"
    )
    print(hdr)
    print("─" * len(hdr))

    for label, N, k, coords in configs:
        nodes = [_MockNode(coords[i]) for i in range(N)]
        d_nn, sigma_D = _mean_nn_distance(nodes)

        r = float(np.mean(np.std(coords, axis=0))) * 1.0 + 1e-9

        temps = {
            "boltzmann": T_boltzmann(nodes, k),
            "rem":        T_rem(nodes, k),
            "noureldin":  T_noureldin(N, r),
        }

        for method, T in temps.items():
            p1 = math.exp(-d_nn   / T)
            p2 = math.exp(-2*d_nn / T)
            sel_str = (
                f"{p1/p2:10.2f}x" if p2 > 1e-30 else "       ∞ (degenerate)"
            )
            print(
                f"{label:35s} {method:12s}"
                f" {T:9.3f} {p1:10.4f} {p2:11.4f} {sel_str}"
            )
        print()

    print("Legend")
    print("  T          : quench temperature in problem coordinate units")
    print("  P(d_nn)    : binding probability at mean nearest-neighbour distance")
    print("  P(2*d_nn)  : binding probability at twice that distance")
    print("  Selectivity: ratio P(d_nn) / P(2*d_nn) — higher = sharper clusters")
    print()
    print("Physical interpretation")
    print("  Boltzmann: P(d_nn) = 1/k_cluster exactly by construction.")
    print("             Selectivity = k_cluster. Sharp, scale-invariant clustering.")
    print("  REM:       Warmer start; useful for wide distance distributions.")
    print("             Lower selectivity; closer to a soft-assignment regime.")
    print("  Noureldin: Scale-blind (T ≈ 0.7–1.2 regardless of coordinate units).")
    print("             P(d_mean) ≈ 0 for all but tiny-coordinate domains.")
    print("             Survives via softmax normalisation (degrades to hard-NN).")
    print("             Retained for regression benchmark only.")


if __name__ == "__main__":
    np.random.seed(42)
    _run_benchmark()
