"""
Least-Action Propagation Network (LAPN) Implementation
=====================================================

A generic, field-based algorithm that materializes the "path of least resistance" principle.
Works for continuous spaces, graphs, and can be implemented in parallel.

Based on the principle that a wavefront emanating from a source obeys Fermat's principle
and the eikonal equation: |∇T(x)| = c(x), with T(source) = 0

The minimal-energy path between two points strictly follows ∇T.
Multiple sources expanding simultaneously create a globally optimal Steiner tree at collision points.
"""

import numpy as np
import heapq
from typing import List, Tuple, Dict, Set, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches


class CellType(Enum):
    FREE = 0
    OBSTACLE = 1
    SOURCE = 2
    TARGET = 3


@dataclass
class SteinerNode:
    """Represents a collision point between two wavefronts."""
    position: Tuple[int, int]
    time: float
    origin_a: int
    origin_b: int

    def __hash__(self):
        return hash(self.position)

    def __eq__(self, other):
        return self.position == other.position


class LAPN:
    """
    Least-Action Propagation Network (LAPN) solver.

    Solves the minimum-action path and Steiner tree problems using
    multi-source wavefront expansion (Fast Marching method).
    """

    def __init__(self,
                 domain_shape: Tuple[int, int],
                 cost_function: np.ndarray,
                 connectivity: int = 4):
        """
        Initialize LAPN solver.

        Args:
            domain_shape: Shape of the grid (rows, cols)
            cost_function: 2D array of cost values (energy per unit step) for each cell
            connectivity: 4 (cardinal) or 8 (including diagonals)
        """
        self.shape = domain_shape
        self.cost = cost_function.copy()
        self.connectivity = connectivity

        # State arrays
        self.T = np.full(domain_shape, np.inf)  # arrival time / action
        self.origin = np.full(domain_shape, -1, dtype=int)  # source region label
        self.predecessor = np.full(domain_shape, -1, dtype=int)  # for backtracking

        # Results
        self.steiner_nodes: List[SteinerNode] = []
        self.edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

        # Neighbors cache
        self._neighbors_cache = {}

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions."""
        if pos in self._neighbors_cache:
            return self._neighbors_cache[pos]

        row, col = pos
        neighbors = []

        # Cardinal directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.connectivity == 8:
            directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.shape[0] and 0 <= nc < self.shape[1]:
                neighbors.append((nr, nc))

        self._neighbors_cache[pos] = neighbors
        return neighbors

    def get_step_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """
        Calculate cost of moving from one cell to another.

        For diagonal moves, cost is multiplied by sqrt(2).
        """
        fr, fc = from_pos
        tr, tc = to_pos

        base_cost = (self.cost[fr, fc] + self.cost[tr, tc]) / 2.0

        # Diagonal movement costs more
        if abs(fr - tr) + abs(fc - tc) == 2:
            base_cost *= np.sqrt(2)

        return base_cost

    def initialize_sources(self, sources: List[Set[Tuple[int, int]]]):
        """
        Initialize source regions.

        Args:
            sources: List of sets, each set contains positions belonging to a source region
        """
        self.T.fill(np.inf)
        self.origin.fill(-1)
        self.predecessor.fill(-1)
        self.steiner_nodes.clear()
        self.edges.clear()

        self.priority_queue = []
        self.in_queue = set()

        for source_id, source_region in enumerate(sources):
            for pos in source_region:
                r, c = pos
                self.T[r, c] = 0.0
                self.origin[r, c] = source_id
                heapq.heappush(self.priority_queue, (0.0, pos))
                self.in_queue.add(pos)

    def expand_wavefront(self, sources: List[Set[Tuple[int, int]]],
                        detect_collisions: bool = True):
        """
        Perform multi-source wavefront expansion (Fast Marching / Dijkstra).

        Args:
            sources: List of source regions
            detect_collisions: If True, detect and record Steiner points
        """
        self.initialize_sources(sources)

        while self.priority_queue:
            current_time, current_pos = heapq.heappop(self.priority_queue)

            if current_pos not in self.in_queue:
                continue
            self.in_queue.remove(current_pos)

            cr, cc = current_pos

            # Skip if we've found a better path already
            if current_time > self.T[cr, cc]:
                continue

            for neighbor in self.get_neighbors(current_pos):
                nr, nc = neighbor

                # Skip obstacles (infinite cost)
                if self.cost[nr, nc] >= 1e10:
                    continue

                # Calculate new arrival time via current position
                step_cost = self.get_step_cost(current_pos, neighbor)
                new_time = self.T[cr, cc] + step_cost

                # Collision detection: different wavefronts meeting
                if detect_collisions and self.origin[nr, nc] >= 0:
                    if self.origin[nr, nc] != self.origin[cr, cc]:
                        # Different regions are meeting
                        if new_time < self.T[nr, nc]:
                            # Record Steiner node at the meeting point
                            steiner = SteinerNode(
                                position=neighbor,
                                time=new_time,
                                origin_a=self.origin[nr, nc],
                                origin_b=self.origin[cr, cc]
                            )
                            self.steiner_nodes.append(steiner)

                # Relaxation step
                if new_time < self.T[nr, nc]:
                    self.T[nr, nc] = new_time
                    self.origin[nr, nc] = self.origin[cr, cc]
                    self.predecessor[nr, nc] = cr * self.shape[1] + cc

                    if neighbor not in self.in_queue:
                        heapq.heappush(self.priority_queue, (new_time, neighbor))
                        self.in_queue.add(neighbor)

    def trace_path(self, target: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Trace back the minimum-action path from target to source.

        Args:
            target: Target position

        Returns:
            List of positions from source to target
        """
        path = []
        current = target

        visited = set()

        while current is not None:
            if current in visited:
                break
            visited.add(current)

            path.append(current)

            r, c = current
            pred_idx = self.predecessor[r, c]

            if pred_idx < 0:
                break

            pred_r = pred_idx // self.shape[1]
            pred_c = pred_idx % self.shape[1]

            # Always move to predecessor
            current = (pred_r, pred_c)

            # Check if we've reached a source region (a cell that is itself a source)
            if self.predecessor[pred_r, pred_c] < 0:
                # This predecessor is a source cell, add it and stop
                path.append(current)
                break

        path.reverse()
        return path

    def build_steiner_tree(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Construct the minimum Steiner tree from Steiner nodes.

        Returns:
            List of edges (pairs of positions) forming the Steiner tree
        """
        edges = []

        for steiner in self.steiner_nodes:
            pos = steiner.position
            r, c = pos

            # Trace back to both source regions
            trace_pos = pos

            while True:
                pred_idx = self.predecessor[r, c]
                if pred_idx < 0:
                    break

                pred_r = pred_idx // self.shape[1]
                pred_c = pred_idx % self.shape[1]

                edges.append(((r, c), (pred_r, pred_c)))

                r, c = pred_r, pred_c

                # Stop if we reach a source or different region
                if self.origin[r, c] != self.origin[pos[0], pos[1]]:
                    break

                if (r, c) == pos:
                    break

        # Remove duplicate edges and simple cycles
        unique_edges = set()
        for e in edges:
            # Normalize edge direction
            if e[0] > e[1]:
                e = (e[1], e[0])
            unique_edges.add(e)

        self.edges = list(unique_edges)
        return self.edges

    def solve_path(self,
                   source_regions: List[Set[Tuple[int, int]]],
                   find_steiner: bool = False) -> Dict:
        """
        Solve the LAPN problem.

        Args:
            source_regions: List of source regions
            find_steiner: If True, compute Steiner tree

        Returns:
            Dictionary with results
        """
        self.expand_wavefront(source_regions, detect_collisions=find_steiner)

        result = {
            'T': self.T.copy(),
            'origin': self.origin.copy(),
            'sources': source_regions
        }

        if find_steiner and len(source_regions) > 1:
            self.build_steiner_tree()
            result['steiner_nodes'] = self.steiner_nodes
            result['edges'] = self.edges

        return result

    @staticmethod
    def create_grid_cost(shape: Tuple[int, int],
                        obstacles: List[Tuple[int, int]] = None,
                        cost_multiplier: float = 1.0) -> np.ndarray:
        """
        Create a cost grid with obstacles.

        Args:
            shape: Grid shape
            obstacles: List of obstacle positions
            cost_multiplier: Base cost multiplier

        Returns:
            Cost array
        """
        cost = np.ones(shape) * cost_multiplier

        if obstacles:
            for r, c in obstacles:
                if 0 <= r < shape[0] and 0 <= c < shape[1]:
                    cost[r, c] = 1e10  # Effectively infinite

        return cost

    @staticmethod
    def create_variable_cost(shape: Tuple[int, int],
                            terrain_func: Callable[[int, int], float]) -> np.ndarray:
        """
        Create a cost grid using a terrain function.

        Args:
            shape: Grid shape
            terrain_func: Function f(row, col) -> cost

        Returns:
            Cost array
        """
        cost = np.zeros(shape)
        for r in range(shape[0]):
            for c in range(shape[1]):
                cost[r, c] = terrain_func(r, c)
        return cost


if __name__ == "__main__":
    print("LAPN module loaded. Use the examples in separate files.")
    print("Run: python lapn_examples.py for demonstrations")
