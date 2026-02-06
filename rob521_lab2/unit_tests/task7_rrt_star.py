# Imports
import numpy as np
from nodes.l2_planning import PathPlanner, Node

### Set up Path Planner
# map info
map_filename = "willowgarageworld_05res.png"
map_setings_filename = "willowgarageworld_05res.yaml"
# robot information
goal_point = np.array([[42], [-44]])  # m
stopping_dist = 0.5  # m
# create planner
path_planner = PathPlanner(
    map_filename, map_setings_filename, goal_point, stopping_dist
)

### RRT_PLANNING
path_planner.rrt_star_planning()
node_path_metric = np.hstack(path_planner.recover_path())

# Leftover test functions
np.save("shortest_path_rrt_star.npy", node_path_metric)