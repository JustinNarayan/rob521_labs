# Imports
import numpy as np
from lab2.nodes.l2_planning import PathPlanner, Node

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
trajectory = path_planner.trajectory_rollout(v=0, w=0, theta_0=0.25*np.pi)
'''
Robot is stationary
'''
print(trajectory)

# Linear velocity
trajectory = path_planner.trajectory_rollout(v=0.5, w=0, theta_0=0.75*np.pi, num_timesteps=4, t_horizon=1)
'''
Robot moves in the -X and +Y direction (left, up in the robot frame)
'''
print(trajectory)

# Rotational velocity
trajectory = path_planner.trajectory_rollout(v=0, w=-0.25, theta_0=0, num_timesteps=4, t_horizon=1)
'''
Robot rotates in CW direction (rotation from left to down)
'''
print(trajectory)

# Combined velocity
trajectory = path_planner.trajectory_rollout(v=0.05, w=0.25, theta_0=0.5*np.pi, num_timesteps=4, t_horizon=2)
'''
Robot start moving straight up and slightly rotates to the left
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

# ...
# path_planner.simulate_trajectory( 0, np.array([0,0]) )
