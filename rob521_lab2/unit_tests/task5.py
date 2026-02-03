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


### Cost to come

# Straight along x
trajectory = np.array([
    [0.00, 0.25, 0.5, 0.75, 1],
    [0,    0,    0,   0,    0],
    [0,    0,    0,   0,    0]
])
cost = path_planner.cost_to_come(trajectory)
'''
Should be some baseline.
'''
print(cost)

# Same along y
trajectory = np.array([
    [0,    0,     0,     0,     0],
    [0,   -0.25, -0.50, -0.75, -1.00],
    [0,    0,     0,     0,     0]
])
cost = path_planner.cost_to_come(trajectory)
'''
Should be same.
'''
print(cost)

# Shorter along y
trajectory = np.array([
    [0,    0,     0,     0,     0],
    [0,   -0.125, -0.25, -0.375, -0.5],
    [0,    0,     0,     0,     0]
])
cost = path_planner.cost_to_come(trajectory)
'''
Should be less.
'''
print(cost)

# Straight along x with slight curve
trajectory = np.array([
    [0.00, 0.25, 0.5, 0.75, 1],
    [0,    0,    0,   0.1,  0.2],
    [0,    0,    0,   0.05, 0.1]
])
cost = path_planner.cost_to_come(trajectory)
'''
Should be larger than the straight path.
'''
print(cost)

# 90 degree rotation
trajectory = np.array([
    [5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5],
    [4/8*np.pi, 5/8*np.pi, 6/8*np.pi, 7/8*np.pi, 8/8*np.pi]
])
cost = path_planner.cost_to_come(trajectory)
'''
Should be some number.
'''
print(cost)

# 180 degree rotation
trajectory = np.array([
    [5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5],
    [0/8*np.pi, 2/8*np.pi, 4/8*np.pi, 6/8*np.pi, 8/8*np.pi]
])
cost = path_planner.cost_to_come(trajectory)
'''
Should be more.
'''
print(cost)


'''

NOTE: It's kind of hard to test the "smoothness" term like this.

It seems like the costs are reasonable, so we can just tune that cost gain.

'''