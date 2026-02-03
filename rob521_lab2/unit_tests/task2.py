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


### Trajectory Rollout

# No velocity
print('no velocity')
trajectory = path_planner.trajectory_rollout(v=0, w=0, theta_0=0.25*np.pi)
'''
Robot is stationary
'''
print(trajectory)

# Linear velocity
print('linear velocity')
trajectory = path_planner.trajectory_rollout(v=0.5, w=0, theta_0=0.75*np.pi, num_timesteps=4, t_horizon=1)
'''
Robot moves in the -X and +Y direction (left, up in the robot frame)
'''
print(trajectory)

# Rotational velocity
print('rotational velocity')
trajectory = path_planner.trajectory_rollout(v=0, w=-0.25, theta_0=0, num_timesteps=4, t_horizon=1)
'''
Robot rotates in CW direction (rotation from left to down)
'''
print(trajectory)

# Combined velocity
print('combined velocity')
trajectory = path_planner.trajectory_rollout(v=0.05, w=0.25, theta_0=0.5*np.pi, num_timesteps=4, t_horizon=2)
'''
Robot start moving straight up and slightly rotates to the left
'''
print(trajectory)

# Starting node
print('starting node')
node_init = Node([5, 5, np.pi/4], 0, 0)
trajectory = path_planner.trajectory_rollout(v=0.05, w=0.25, theta_0=0.5*np.pi, num_timesteps=4, t_horizon=2, starting_node=node_init)
'''
Robot start moving straight up and slightly rotates to the left, but from (5, 5, 45 degrees)
'''
print(trajectory)


### Robot Controller

# Straight ahead
node_i = Node( (0,0,np.pi), 0, 0 )
v,w = path_planner.robot_controller(
    node_i, (-5, 0)
)
'''
New point to the left and robot already facing direction.
Expect only linear velocity.
'''
print(v, w)

# Rotation -ve
v,w = path_planner.robot_controller(
    node_i, (5, 0.5)
)
'''
Robot facing left. Point is to the right and slightly up.
Expect full rotation CW.
'''
print(v, w)

# Rotation +ve
v,w = path_planner.robot_controller(
    node_i, (5, -0.5)
)
'''
Robot facing left. Point is to the right and slightly down.
Expect full rotation CCW.
'''
print(v, w)

# Mix rotation and velocity
v,w = path_planner.robot_controller(
    node_i, (-10, 1)
)
'''
Robot facing left. Point is further right and slightly up.
Expect some forward and CCW movement.
'''
print(v,w)


### Simulate Trajectory 

# Path 1
node_i = Node( (0,0,0), 0, 0 )
trajectory = path_planner.simulate_trajectory(node_i, (10,0))
'''
Should be valid increasing in X.
'''
print(trajectory)

# Path 2
node_i = Node( (10,78.8,np.pi/2), 0, 0 )
trajectory = path_planner.simulate_trajectory(node_i, (10,100))
'''
Should be valid decreasing in Y.
'''
print(trajectory)

# Path 3
node_i = Node( (10,78.8,np.pi/2), 0, 0 )
trajectory = path_planner.simulate_trajectory(node_i, (9,0))
'''
Should be rotating from pi (up) CCW to down-left (increasing theta).
'''
print(trajectory)


### Collision Checks
# Arbitrary occupancy map. 1 = Free, 0 = Black
path_planner.occupancy_map = np.array([
    # y is indexed first for the row, x is indexed second for the col
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
])

# Cell Collision checks
print('free cell')
free = path_planner.cell_collision_free(np.array([ [0], [0] ]))
'''
Free cell. Return 1.
'''
print(free)

print('free cell')
free = path_planner.cell_collision_free(np.array([ [2], [1] ]))
'''
Free cell. Return 1.
'''
print(free)

print('colliding cell')
free = path_planner.cell_collision_free(np.array([ [1], [2] ]))
'''
Collision cell. Return 0.
'''
print(free)

print('colliding cell')
free = path_planner.cell_collision_free(np.array([ [8], [8] ]))
'''
Collision cell. Return 0.
'''
print(free)

# Multi-cell Collision checks
print('one free set')
free_cells = path_planner.cells_collision_free(np.array([
    # Set 1
    [
        [0, 1, 2], # x
        [0, 0, 0] # y
    ]
]))
'''
Free cells.
'''
print(free_cells)

print('two free sets')
free_cells = path_planner.cells_collision_free(np.array([
    # Set 1
    [
        [2, 3, 3], # x
        [1, 1, 2]  # y
    ],
    # Set 2
    [
        [0, 1, 2], # x 
        [7, 7, 7]  # y
    ]
]))
'''
Free cells.
'''
print(free_cells)

print('two colliding sets, all colliding')
free_cells = path_planner.cells_collision_free(np.array([
    # Set 1
    [
        [1, 1, 2], # x
        [1, 2, 3]  # y
    ],
    # Set 2
    [
        [7, 8, 9], # x 
        [8, 8, 8]  # y
    ]
]))
'''
All colliding cells.
'''
print(free_cells)

print('two colliding sets, mix of free and colliding')
free_cells = path_planner.cells_collision_free(np.array([
    # Set 1
    [
        [0, 1, 2], # x
        [0, 0, 3]  # y
    ],
    # Set 2
    [
        [0, 1, 9], # x 
        [8, 8, 8]  # y
    ]
]))
'''
Mix of free and colliding cells.
'''
print(free_cells)

print('one colliding set and one free set.')
free_cells = path_planner.cells_collision_free(np.array([
    # Set 1
    [
        [0, 1, 2], # x
        [0, 0, 3]  # y
    ],
    # Set 2
    [
        [0, 1, 2], # x 
        [8, 8, 8]  # y
    ]
]))
'''
First set is a mix of colliding and free cells. Second set is free cells.
'''
print(free_cells)