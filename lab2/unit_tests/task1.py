# Stop pygame warning
import warnings
warnings.simplefilter("ignore", category=UserWarning)

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


### Point to Cell

# ...
path_planner.point_to_cell([0,0])

# ...
path_planner.point_to_cell([1,1])



### Points to Robot Circle

# ...
path_planner.points_to_robot_circle([ [0,0], [1,0] ])
