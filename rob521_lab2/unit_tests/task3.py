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


### Sample map space

# full map
print('full map')
coords = path_planner.sample_map_space()
'''
Within full map. Should be between 0 (inclusive) and 1600 (exclusive) for both
'''
print(coords)

# subset
print('subset')
coords = path_planner.sample_map_space(subset_x=[500,510], subset_y=[300,310])
'''
Within subset
'''
print(coords)

### Check if duplicate
path_planner.nodes = [
    Node([5, 4, 0], 0, 0, 0),
    Node([10, 3, 30], 1, 0, 0)
]

# new node
print('new node')
is_dupe = path_planner.check_if_duplicate([8, 6])
'''
Should be false.
'''
print(is_dupe)

# new node, same x
print('new node, same x')
is_dupe = path_planner.check_if_duplicate([5, 6])
'''
Should be false.
'''
print(is_dupe)

# new node, same y
print('new node, same x')
is_dupe = path_planner.check_if_duplicate([5, 3])
'''
Should be false.
'''
print(is_dupe)

# repeat node, same theta
print('repeat node')
is_dupe = path_planner.check_if_duplicate([10, 3])
'''
Should be true.
'''
print(is_dupe)

### Closest node
path_planner.nodes = [] # empty

# no nodes
print('no nodes')
closest_id = path_planner.closest_node([6, 30])
'''
Should be None, no nodes.
'''
print(closest_id)

# one node
path_planner.nodes.append(Node([3, 3, 0], 0, -1, 0))
print('one node')
closest_id = path_planner.closest_node([6, 30])
'''
Should be 0, only node.
'''
print(closest_id)

# a closer node
path_planner.nodes.append(Node([6, 29, 0], 1, 0, 0))
print('a closer node')
closest_id = path_planner.closest_node([6, 30])
'''
Should be 1, closer node.
'''
print(closest_id)

# more nodes, arbitrary
path_planner.nodes.append(Node([5, 28, 0], 2, 1, 0))
path_planner.nodes.append(Node([7, 31, 0], 3, 1, 0))
path_planner.nodes.append(Node([6.5, 29.5, 0], 4, 3, 0))
print('more nodes, arbitrary')
closest_id = path_planner.closest_node([6, 30])
'''
Should be 4, closest node.
'''
print(closest_id)