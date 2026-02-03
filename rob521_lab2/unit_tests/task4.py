# Imports
import numpy as np
from rob521_lab2.nodes.l2_planning import PathPlanner, Node

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


### Connect node to point
node_i = Node([ 5, 5, np.pi/2], 0, 0) # facing up

# Straight ahead
print('straight ahead')
traj = path_planner.connect_node_to_point(node_i, [5, 6.12]) # straight up, already facing up
'''
Trajectory should be no rotation and then vertical movement.
'''
print(traj)

# Directly behind
print('straight behind')
traj = path_planner.connect_node_to_point(node_i, [5, 3]) # straight down, already facing up
'''
Trajectory should be a 180 degree rotation and then vertical movement.
'''
print(traj)

# Arbitrary position 1
print('up and left')
traj = path_planner.connect_node_to_point(node_i, [3, 8]) # up and left.
'''
Trajectory should be a slight CCW (icnreasing theta) rotation and then movement up and left.
'''
print(traj)

# Arbitrary position 2
print('down and right')
traj = path_planner.connect_node_to_point(node_i, [7, 5.25]) # down and right.
'''
Trajectory should be a slight CW (decreasing theta) rotation and then movement down and right.
'''
print(traj)