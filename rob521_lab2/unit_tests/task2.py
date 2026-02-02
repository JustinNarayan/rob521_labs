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
node_init = Node([5, 5, np.pi/4], 0, 0, 0)
trajectory = path_planner.trajectory_rollout(v=0.05, w=0.25, theta_0=0.5*np.pi, num_timesteps=4, t_horizon=2, starting_node=node_init)
'''
Robot start moving straight up and slightly rotates to the left, but from (5, 5, 45 degrees)
'''
print(trajectory)


### Robot Controller

# Straight ahead
node_i = Node( (0,0,np.pi), 0, 0, 0 )
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
# IN PROGRESS

# Free path 1
node_i = Node( (0,0,0), 0, 0, 0 )
trajectory = path_planner.simulate_trajectory(node_i, (10,0))
'''
For this map, this is a free path. Should be valid increasing in X.
'''
print(trajectory)

# Free path 2
node_i = Node( (10,78.8,np.pi/2), 0, 0, 0 )
trajectory = path_planner.simulate_trajectory(node_i, (10,100))
'''
For this map, this is a free path. Should be valid decreasing in Y.
'''
print(trajectory)

# Free path 3
node_i = Node( (10,78.8,np.pi/2), 0, 0, 0 )
trajectory = path_planner.simulate_trajectory(node_i, (9,0))
'''
For this map, this is a free path. Should be rotating from pi (up) CCW to down-left (increasing theta).
'''
print(trajectory)

# Collision path
node_i = Node( (61,78.6,np.pi/2), 0, 0, 0 )
trajectory = path_planner.simulate_trajectory(node_i, (61,100))
'''
For this map, this is a free path. Should be rotating from pi (up) CCW to down-left (increasing theta).
'''
print(trajectory)