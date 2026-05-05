"""
LAPN Algorithm Examples and Demonstrations
==========================================

Demonstrates various applications of the Least-Action Propagation Network:
1. Pathfinding with obstacles and terrain
2. Steiner tree connecting multiple source regions
3. Power grid topology design
4. Neural network implementation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from lapn_algorithm import LAPN


def example1_pathfinding():
    """
    Example 1: Pathfinding on 2D grid with obstacles and varying terrain.

    Demonstrates minimum-action path finding where cost varies across space,
    simulating terrain elevation, obstacles, or resistance.
    """
    print("=" * 60)
    print("Example 1: Pathfinding with Terrain and Obstacles")
    print("=" * 60)

    # Grid setup
    rows, cols = 40, 60

    # Create terrain with varying cost
    # Add "mountains" (high cost) and "valleys" (low cost)
    def terrain_func(r, c):
        base = 1.0
        # Add hills
        hill1 = 5.0 * np.exp(-((r - 15)**2 + (c - 20)**2) / 100.0)
        hill2 = 4.0 * np.exp(-((r - 25)**2 + (c - 45)**2) / 80.0)
        # Add river (low cost path)
        river = 0.3 * np.exp(-((c - 30)**2) / 50.0)
        return base + hill1 + hill2 - river

    cost = LAPN.create_variable_cost((rows, cols), terrain_func)

    # Add obstacles
    obstacles = []
    for r in range(10, 20):
        obstacles.append((r, 35))
    for c in range(20, 30):
        obstacles.append((15, c))
    for r in range(30, 38):
        obstacles.append((r, 50))

    for r, c in obstacles:
        cost[r, c] = 1e10

    # Define source and target
    source_pos = (5, 5)
    target_pos = (35, 55)
    source_region = {source_pos}
    target_region = {target_pos}

    # Solve: Single source, target is a point to find path to
    lapn = LAPN((rows, cols), cost, connectivity=8)
    result = lapn.solve_path([source_region], find_steiner=False)

    # Trace path from target back to source
    path = lapn.trace_path(target_pos)

    # Calculate total cost along the path
    path_cost = result['T'][target_pos] if result['T'][target_pos] < 1e9 else np.inf

    # Print results
    print(f"Grid size: {rows}x{cols}")
    print(f"Source: {source_pos}")
    print(f"Target: {target_pos}")
    print(f"Path length: {len(path)} cells")
    print(f"Total action (cost): {path_cost:.2f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Terrain cost
    ax = axes[0]
    im = ax.imshow(cost, cmap='terrain', origin='upper', vmin=0, vmax=6)
    ax.set_title('Terrain Cost (darker = higher cost)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Cost')

    # Mark obstacles
    if obstacles:
        obs_r, obs_c = zip(*obstacles)
        ax.scatter(obs_c, obs_r, c='red', s=20, marker='s', label='Obstacles', alpha=0.7)

    # Plot 2: Arrival time field
    ax = axes[1]
    T_display = result['T'].copy()
    T_display[T_display > 1000] = np.nan
    im = ax.imshow(T_display, cmap='hot', origin='upper')
    ax.set_title('Arrival Time Field (Fast Marching)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Time/Action')

    # Mark source and target
    ax.scatter(source_pos[1], source_pos[0], c='green', s=100, marker='o', label='Source', zorder=5)
    ax.scatter(target_pos[1], target_pos[0], c='blue', s=100, marker='s', label='Target', zorder=5)
    ax.legend()

    # Plot 3: Path overlay
    ax = axes[2]
    ax.imshow(cost, cmap='terrain', origin='upper', alpha=0.5, vmin=0, vmax=6)
    ax.set_title('Optimal Path (Least Action Principle)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    if path and len(path) > 1:
        path_r, path_c = zip(*path)
        ax.plot(path_c, path_r, 'r-', linewidth=2.5, label='Optimal Path', alpha=0.9)
        ax.scatter(path_c, path_r, c='red', s=20, alpha=0.5)

    ax.scatter(source_pos[1], source_pos[0], c='green', s=100, marker='o', label='Source', zorder=5)
    ax.scatter(target_pos[1], target_pos[0], c='blue', s=100, marker='s', label='Target', zorder=5)
    ax.legend()

    plt.tight_layout()
    plt.savefig('example1_pathfinding.png', dpi=150, bbox_inches='tight')
    print("Saved: example1_pathfinding.png\n")
    plt.close()

    return result, path


def example2_steiner_tree():
    """
    Example 2: Minimum Steiner tree connecting multiple source regions.

    Demonstrates how multiple wavefronts expanding simultaneously create
    a globally optimal network connecting all sources at Steiner points.
    """
    print("=" * 60)
    print("Example 2: Steiner Tree (Multiple Source Regions)")
    print("=" * 60)

    rows, cols = 40, 40

    # Uniform cost
    cost = np.ones((rows, cols))

    # Add some obstacles
    for c in range(15, 25):
        cost[20, c] = 1e10
    for r in range(10, 20):
        cost[r, 10] = 1e10

    # Define multiple source regions (clusters)
    source1 = {(5, 5), (5, 6), (6, 5), (6, 6)}
    source2 = {(5, 35), (5, 34), (6, 35), (6, 34)}
    source3 = {(35, 5), (35, 6), (34, 5), (34, 6)}
    source4 = {(35, 35), (35, 34), (34, 35), (34, 34)}
    source5 = {(15, 30), (16, 30)}

    sources = [source1, source2, source3, source4, source5]

    # Solve with Steiner tree
    lapn = LAPN((rows, cols), cost, connectivity=8)
    result = lapn.solve_path(sources, find_steiner=True)

    # Print results
    print(f"Grid size: {rows}x{cols}")
    print(f"Number of source regions: {len(sources)}")
    print(f"Steiner nodes found: {len(result['steiner_nodes'])}")
    print(f"Edges in Steiner tree: {len(result['edges'])}")

    total_cost = 0
    for (r1, c1), (r2, c2) in result['edges']:
        step_cost = lapn.get_step_cost((r1, c1), (r2, c2))
        total_cost += step_cost
    print(f"Total tree cost: {total_cost:.2f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Arrival time field
    ax = axes[0]
    T_display = result['T'].copy()
    T_display[T_display > 1000] = np.nan
    im = ax.imshow(T_display, cmap='viridis', origin='upper')
    ax.set_title('Multi-Source Wavefront Expansion')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Time')

    colors = ['lime', 'cyan', 'yellow', 'magenta', 'orange']
    for i, src in enumerate(sources):
        src_list = list(src)
        r, c = zip(*src_list)
        ax.scatter(c, r, c=colors[i], s=50, label=f'Region {i+1}', zorder=5)

    ax.legend()

    # Plot 2: Origin field (which source each cell belongs to)
    ax = axes[1]
    origin_display = result['origin'].astype(float)
    origin_display[origin_display < 0] = np.nan
    im = ax.imshow(origin_display, cmap='tab10', origin='upper', vmin=0, vmax=9)
    ax.set_title('Source Region Assignment')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Source Region', ticks=range(len(sources)))

    # Plot 3: Steiner tree
    ax = axes[2]
    ax.imshow(cost, cmap='gray', origin='upper', alpha=0.3)
    ax.set_title('Minimum Steiner Tree (Least-Action Network)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Draw edges
    for (r1, c1), (r2, c2) in result['edges']:
        ax.plot([c1, c2], [r1, r2], 'red', linewidth=1.5, alpha=0.8)

    # Draw sources
    for i, src in enumerate(sources):
        src_list = list(src)
        r, c = zip(*src_list)
        ax.scatter(c, r, c=colors[i], s=100, marker='s',
                  label=f'Region {i+1}', zorder=5, edgecolors='black', linewidths=0.5)

    # Draw Steiner nodes
    for node in result['steiner_nodes']:
        r, c = node.position
        ax.scatter(c, r, c='red', s=200, marker='*', edgecolors='white',
                  linewidths=1.5, label='Steiner Point', zorder=6)

    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig('example2_steiner_tree.png', dpi=150, bbox_inches='tight')
    print("Saved: example2_steiner_tree.png\n")
    plt.close()

    return result


def example3_power_grid():
    """
    Example 3: Power grid topology design.

    Simulates connecting power plants to consumer clusters with minimal
    resistive losses (cost proportional to distance and terrain).
    """
    print("=" * 60)
    print("Example 3: Power Grid Topology Design")
    print("=" * 60)

    rows, cols = 50, 60

    # Cost represents terrain difficulty + distance
    # Adding a "mountain range" that's expensive to cross
    cost = np.ones((rows, cols))
    for r in range(rows):
        for c in range(cols):
            # Mountain ridge
            mountain_cost = 8.0 * np.exp(-((c - 30)**2) / 40.0)
            # River valley (easy)
            valley_cost = -0.5 * np.exp(-((c - 45)**2 + (r - 25)**2) / 200.0)
            cost[r, c] = 1.0 + max(0, mountain_cost + valley_cost)

    # Power plants (sources)
    plant1 = {(45, 10)}
    plant2 = {(5, 55)}

    # Consumer clusters (targets) - demand centers
    urban1 = {(r, c) for r in range(8, 13) for c in range(18, 23)}
    urban2 = {(r, c) for r in range(35, 40) for c in range(45, 50)}
    urban3 = {(r, c) for r in range(20, 25) for c in range(25, 30)}

    sources = [plant1, plant2, urban1, urban2, urban3]

    # Solve
    lapn = LAPN((rows, cols), cost, connectivity=8)
    result = lapn.solve_path(sources, find_steiner=True)

    print(f"Grid size: {rows}x{cols}")
    print(f"Power plants: 2")
    print(f"Urban clusters: 3")
    print(f"Total regions to connect: {len(sources)}")
    print(f"Steiner nodes (substations): {len(result['steiner_nodes'])}")
    print(f"Transmission lines: {len(result['edges'])}")

    total_cost = 0
    for (r1, c1), (r2, c2) in result['edges']:
        step_cost = lapn.get_step_cost((r1, c1), (r2, c2))
        total_cost += step_cost
    print(f"Total infrastructure cost: {total_cost:.2f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Terrain/cost
    ax = axes[0]
    im = ax.imshow(cost, cmap='terrain', origin='upper')
    ax.set_title('Grid Cost (Mountains, Valleys)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Cost')

    # Plot 2: Arrival time from all sources
    ax = axes[1]
    T_display = result['T'].copy()
    T_display[T_display > 1000] = np.nan
    im = ax.imshow(T_display, cmap='plasma', origin='upper')
    ax.set_title('Network Propagation Time')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Time')

    for src in sources[:2]:
        r, c = list(src)[0]
        ax.scatter(c, r, c='yellow', s=200, marker='*', label='Plant' if src == sources[0] else '', zorder=5)

    for src in sources[2:]:
        src_list = list(src)
        r_vals, c_vals = zip(*src_list)
        ax.scatter(c_vals, r_vals, c='red', s=30, alpha=0.5)

    # Plot 3: Power grid network
    ax = axes[2]
    ax.imshow(cost, cmap='terrain', origin='upper', alpha=0.4)
    ax.set_title('Optimal Power Grid (Steiner Tree)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Draw edges
    for (r1, c1), (r2, c2) in result['edges']:
        ax.plot([c1, c2], [r1, r2], 'yellow', linewidth=2, alpha=0.8)

    # Power plants
    for src in sources[:2]:
        r, c = list(src)[0]
        ax.scatter(c, r, c='yellow', s=300, marker='*', edgecolors='black',
                  linewidths=1, label='Power Plant', zorder=5)

    # Urban clusters
    for src in sources[2:]:
        src_list = list(src)
        r_vals, c_vals = zip(*src_list)
        ax.scatter(c_vals, r_vals, c='red', s=50, alpha=0.6, label='Demand Center' if src == sources[2] else '', zorder=4)

    # Substations (Steiner nodes)
    for node in result['steiner_nodes']:
        r, c = node.position
        ax.scatter(c, r, c='orange', s=150, marker='s', edgecolors='white',
                  linewidths=1.5, label='Substation' if node == result['steiner_nodes'][0] else '', zorder=6)

    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig('example3_power_grid.png', dpi=150, bbox_inches='tight')
    print("Saved: example3_power_grid.png\n")
    plt.close()

    return result


def example4_neural_network_layer():
    """
    Example 4: Differentiable LAPN layer (neural network).

    Shows how LAPN can be implemented as a differentiable layer
    using gradient computation through the unrolled dynamics.
    This is a simplified demonstration - full implementation
    would use PyTorch/TensorFlow autograd.
    """
    print("=" * 60)
    print("Example 4: Neural Network LAPN Layer (Differentiable)")
    print("=" * 60)

    rows, cols = 20, 20

    # Learnable cost map (parameterized)
    np.random.seed(42)
    base_cost = np.ones((rows, cols))

    # Add some learned features (simulating a trained cost map)
    # This would normally be learned from data
    for i in range(100):
        center_r = np.random.randint(0, rows)
        center_c = np.random.randint(0, cols)
        strength = np.random.uniform(0.5, 3.0)
        width = np.random.uniform(5, 15)
        for r in range(rows):
            for c in range(cols):
                dist2 = (r - center_r)**2 + (c - center_c)**2
                base_cost[r, c] += strength * np.exp(-dist2 / width)

    # Normalize
    base_cost = base_cost / base_cost.max() * 2.0 + 0.1

    # Sources: start and goal
    source = {(2, 2)}
    target = {(17, 17)}
    sources = [source, target]

    # Solve multiple times with slightly perturbed costs
    # to simulate gradient computation
    lapn = LAPN((rows, cols), base_cost, connectivity=8)
    result = lapn.solve_path(sources, find_steiner=False)

    path = lapn.trace_path(list(target)[0])

    print(f"Grid size: {rows}x{cols}")
    print(f"Path found: {len(path)} steps")
    print(f"Total cost: {result['T'][list(target)[0]]:.4f}")

    # Simulate gradient: compute sensitivity by perturbing cost
    epsilon = 0.01
    gradients = np.zeros((rows, cols))

    print("\nComputing cost sensitivity (simulated gradient)...")

    # Sample a few cells for gradient estimation
    sample_cells = [(5, 5), (10, 10), (15, 15)]
    for r, c in sample_cells:
        # Perturb cost positively
        cost_plus = base_cost.copy()
        cost_plus[r, c] += epsilon

        lapn_plus = LAPN((rows, cols), cost_plus, connectivity=8)
        result_plus = lapn_plus.solve_path(sources, find_steiner=False)
        cost_plus_val = result_plus['T'][list(target)[0]]

        # Perturb cost negatively
        cost_minus = base_cost.copy()
        cost_minus[r, c] -= epsilon

        lapn_minus = LAPN((rows, cols), cost_minus, connectivity=8)
        result_minus = lapn_minus.solve_path(sources, find_steiner=False)
        cost_minus_val = result_minus['T'][list(target)[0]]

        # Finite difference gradient
        gradient = (cost_plus_val - cost_minus_val) / (2 * epsilon)
        gradients[r, c] = gradient
        print(f"  Cell ({r},{c}): gradient = {gradient:.4f}")

    print(f"\nInterpretation:")
    print("  Negative gradient -> decreasing cost here reduces total path cost")
    print("  Positive gradient -> increasing cost here reduces total path cost")
    print("  (Path seeks to pass through cells with lower cost)")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Learned cost map
    ax = axes[0]
    im = ax.imshow(base_cost, cmap='hot', origin='upper')
    ax.set_title('Learned Cost Map (Neural Network Output)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Cost')

    # Plot 2: Path
    ax = axes[1]
    T_display = result['T'].copy()
    T_display[T_display > 1000] = np.nan
    im = ax.imshow(T_display, cmap='viridis', origin='upper')
    ax.set_title('Arrival Time Field')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Time')

    if path and len(path) > 1:
        path_r, path_c = zip(*path)
        ax.plot(path_c, path_r, 'r-', linewidth=2, label='Optimal Path')

    src_r, src_c = list(source)[0]
    tgt_r, tgt_c = list(target)[0]
    ax.scatter(src_c, src_r, c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(tgt_c, tgt_r, c='blue', s=100, marker='s', label='Goal', zorder=5)
    ax.legend()

    # Plot 3: Gradient sensitivity
    ax = axes[2]
    grad_display = gradients.copy()
    grad_display[grad_display == 0] = np.nan
    im = ax.imshow(base_cost, cmap='gray', origin='upper', alpha=0.3)
    im2 = ax.imshow(grad_display, cmap='RdBu_r', origin='upper', alpha=0.7,
                   vmin=-abs(gradients).max(), vmax=abs(gradients).max())
    ax.set_title('Cost Sensitivity (Gradient)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im2, ax=ax, label='Gradient dLoss/dCost')

    if path and len(path) > 1:
        ax.plot(path_c, path_r, 'k--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig('example4_neural_network.png', dpi=150, bbox_inches='tight')
    print("\nSaved: example4_neural_network.png\n")
    plt.close()

    return result, gradients


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LEAST-ACTION PROPAGATION NETWORK (LAPN)")
    print("Comprehensive Examples")
    print("=" * 60 + "\n")

    # Example 1
    example1_pathfinding()

    # Example 2
    example2_steiner_tree()

    # Example 3
    example3_power_grid()

    # Example 4
    example4_neural_network_layer()

    print("=" * 60)
    print("All examples completed!")
    print("Generated files:")
    print("  - example1_pathfinding.png")
    print("  - example2_steiner_tree.png")
    print("  - example3_power_grid.png")
    print("  - example4_neural_network.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
