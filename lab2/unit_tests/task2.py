# Imports
import numpy as np
from lab2.nodes.l2_planning import PathPlanner

### Set up Path Planner
# map info
map_filename = "willowgarageworld_05res.png"
map_setings_filename = "willowgarageworld_05res.yaml"
# robot information
goal_point = np.array([[10], [10]])  # m
stopping_dist = 0.5  # m
# create planner
path_planner = PathPlanner(
    map_filename, map_setings_filename, goal_point, stopping_dist
)


### Trajectory Rollout

# ...
path_planner.trajectory_rollout( 0.5, 0.1 )

# ...
path_planner.trajectory_rollout( 0.1, 0.2 )



### Robot Controller

# ...
path_planner.robot_controller( 0, np.array([0,0]) )



### Simulate Trajectory

# ...
path_planner.simulate_trajectory( 0, np.array([0,0]) )
