"""Route and map heuristic policies for MuZero.

Keep route scoring, route safety guards, and route-prior bias logic here rather
than in ``muzero.train`` or the generic MCTS implementation.
"""

from .root_bias import compute_route_heuristic_bias_vector

__all__ = ["compute_route_heuristic_bias_vector"]
