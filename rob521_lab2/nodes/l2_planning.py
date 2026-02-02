#!/usr/bin/env python3
# Standard Libraries
import os
import numpy as np
from math import sqrt
import yaml
import pygame
import time
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag

# needed to make this work on Windows
# import pygame_utils
import rob521_labs.lab2.nodes.pygame_utils as pygame_utils

def normalize_angle(angle):
    return np.atan2( np.sin(angle), np.cos(angle) ) # now in [-np.pi, np.pi]

def load_map(filename):
    # Get the filepath
    full_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "lab2", "maps", filename)
    )
    
    # Load
    im = mpimg.imread(full_path)
    if len(im.shape) > 2:
        im = im[:, :, 0]
    im_np = np.array(im)  # Whitespace is true, black is false
    # im_np = np.logical_not(im_np)
    return im_np


def load_map_yaml(filename):
    # Get the filepath
    full_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "lab2", "maps", filename)
    )
    
    # Load
    with open(full_path, "r") as stream:
        map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict


# Node for building a graph
class Node:
    def __init__(self, pose, parent_id, cost):
        self.pose = pose  # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id  # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost  # The cost to come to this node
        self.children_ids = []  # The children node ids of this node
        return

    def get_pose(self):
        return self.pose

# Path Planner
class PathPlanner:
    # A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        # Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        # Get the metric bounds of the map
        self.bounds = np.zeros([2, 2])  # m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = (
            self.map_settings_dict["origin"][0]
            + self.map_shape[1] * self.map_settings_dict["resolution"]
        )
        self.bounds[1, 1] = (
            self.map_settings_dict["origin"][1]
            + self.map_shape[0] * self.map_settings_dict["resolution"]
        )

        # Robot information
        self.robot_radius = 0.22  # m
        self.vel_max = 0.5  # m/s (Feel free to change!)
        self.rot_vel_max = 0.2  # rad/s (Feel free to change!)

        # Goal Parameters
        self.goal_point = goal_point  # m
        self.stopping_dist = stopping_dist  # m

        # Trajectory Simulation Parameters
        self.timestep = 1.0  # s
        self.num_substeps = 10

        # Planning storage
        self.nodes = [Node(np.zeros((3, 1)), -1, 0)]

        # RRT* Specific Parameters
        self.lebesgue_free = (
            np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] ** 2
        )
        self.zeta_d = np.pi
        self.gamma_RRT_star = (
            2 * (1 + 1 / 2) ** (1 / 2) * (self.lebesgue_free / self.zeta_d) ** (1 / 2)
        )
        self.gamma_RRT = self.gamma_RRT_star + 0.1
        self.epsilon = 2.5

        # Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner",
            (1000, 1000),
            self.occupancy_map.shape,
            self.map_settings_dict,
            self.goal_point,
            self.stopping_dist,
        )
        return

    # Functions required for RRT
    def sample_map_space(self):
        # Return an [x,y] coordinate to drive the robot towards
        print("TO DO: Sample point to drive towards")
        return np.zeros((2, 1))

    def check_if_duplicate(self, point):
        # Check if point is a duplicate of an already existing node
        print("TO DO: Check that nodes are not duplicates")
        return False

    def closest_node(self, point):
        # Returns the index of the closest node
        print("TO DO: Implement a method to get the closest node to a sapled point")
        return 0

    def simulate_trajectory(self, node_i: Node, point_s):
        # IN PROGRESS
        #
        # A starting node and goal point is selected.
        # Both are given in the inertial frame.
        # 
        # Linear and rotation velocity commands are chosen for the robot in the global frame.
        # These commands, plus the starting node are passed to the robot.
        # A global output trajectory is produced:
        #   - x,y coordinates in the global frame
        #   - theta heading in the global frame
        #
        # The output trajectory is returned if there are no collisons; otherwise None
        v, w = self.robot_controller(node_i, point_s)
        trajectory = self.trajectory_rollout(v, w, node_i.get_pose()[2], starting_node=node_i)
        
        if self.trajectory_collision_free(trajectory):
            return trajectory
        return None

    def robot_controller(self, node_i: Node, point_s):
        # Node starts from (x_0, y_0, theta_0)
        # Point contains (x,y)
        # Generate a (v, w) pair to push the robot to (x, y, ?) from (x_0, y_0, theta_0)
        
        # Extract points
        x_0, y_0, theta_0 = node_i.get_pose()
        x, y = point_s
        
        # Compute errors
        dX, dY = x - x_0, y - y_0
        dist = sqrt(dX**2 + dY**2)
        theta = np.atan2(dY, dX)
        dTheta = theta - theta_0
        dTheta = normalize_angle(dTheta) # now in [-np.pi, np.pi]
        
        # If large dTheta, just rotate
        min_dTheta_for_just_rotation = (1/3)*np.pi
        v,w = 0,0
        if abs(dTheta) > min_dTheta_for_just_rotation:
            v = 0
            w = self.rot_vel_max * np.sign(dTheta)
        
        # Simple proportional controller
        else:
            K_v, K_w = 0.5, 1.0 # prioritize rotation
            v = np.clip(
                K_v * dist, 0, self.vel_max
            )
            w = np.clip(
                K_w * dTheta, -self.rot_vel_max, self.rot_vel_max
            )
        
        # Return
        return v, w

    def trajectory_rollout(
        self, 
        v, # velocity 
        w, # rotational velocity
        theta_0, 
        num_timesteps = 10, # self.num_substeps 
        t_horizon = 1, # self.timestep
        starting_node: Node = None
    ):
        # Compute a set of waypoints provided a velocity (m/s) and rotational velocity (rad/s)
        # A starting theta is required to compute dX and dY in the global frame.
        # +w is rotation CCW
        #
        # If starting_node is None:
        #   Output dX, dY, dTheta is in the global, inertial frame.
        # Else:
        #   Output X, Y, Theta in the global, intertial frame
        #
        # Compute <num_timesteps> timesteps evenly spaced @ <dt> second increments
        #
        # Unicycle model:
        # |   x_dot   |   | cos(theta) 0 | | v |
        # |   y_dot   | = | sin(theta) 0 | | w |
        # | theta_dot |   |     0      1 |
        #
        # -> x_dot = cos(theta) * v
        # -> y_dot = sin(theta) * v
        # -> theta_dot = w
        #
        # To get timesteps, we need to integrate:
        # 
        # -> dX     = (v/w) ( sin(theta_0 + w*t) - sin(theta_0))
        # -> dY     = (v/w) (-cos(theta_0 + w*t) + cos(theta_0))
        # -> dTheta = w*t
        #
        # If w=0 and it's a straight line:
        #
        # -> dX     = v*t*cos(theta_0)
        # -> dY     = v*t*sin(theta_0)
        # -> dTheta = 0
        #
        # Output trajectory: 3x<num_timesteps>:
        # | [dX1,     dX2,     ...]
        # | [dY1,     dY2,     ...]
        # | [dTheta1, dTheta2, ...]
        
        # Time
        t = np.linspace(0, t_horizon, num_timesteps)
        
        # Compute trajectory
        xs, ys = [], []
        thetas = w * t # same regardless of w
        # w = 0
        if w == 0:
            xs = [ v * t * np.cos(theta_0) ]
            ys = [ v * t * np.sin(theta_0) ]
        # w != 0
        else:
            xs = [(v/w) * ( np.sin(theta_0 + w*t) - np.sin(theta_0) )]
            ys = [(v/w) * (-np.cos(theta_0 + w*t) + np.cos(theta_0) )]
            
        # Account for starting node
        if starting_node is not None:
            x_i, y_i, theta_i = starting_node.get_pose()
            xs = [x + x_i for x in xs]
            ys = [y + y_i for y in ys]
            thetas = [normalize_angle(theta+theta_i) for theta in thetas]
        
        # Return trajectory
        return np.vstack( (xs, ys, thetas) )

    def point_to_cell(self, points):
        # points: a series of (2xN) points of interest from the map reference
        #       | [x1, x2, ..., xN]
        # i.e.  | [y1, y2, ..., yN]
        #
        # Convert each (x,y) pair to the indices in occupancy map
        # The map's "reference" frame is the bottom left-hand corner of the map
        # The occupancy map's reference frame is the true origin
        #
        # Output: cells
        #      | [xmap1, xmap2, ..., xmapN]
        # i.e. | [ymap1, ymap2, ..., ymapN]
        
        # Output
        cells = np.zeros_like(points)
        
        # Extract map properties
        res_m_per_px = self.map_settings_dict['resolution']
        w_px, h_px = self.map_shape
        w_m, h_m = w_px * res_m_per_px, h_px * res_m_per_px
        
        # "Origin" property is offset of map reference frame from the true origin
        # Provided r_PF (point from frame), get r_PO (point from origin)
        # r_PO = r_PF - r_FO (frame from origin)
        frame_from_origin = np.array(self.map_settings_dict['origin'][:2]).reshape(2,1)
        
        # Points are given in meters
        num_points = points.shape[1]
        for i in range(num_points):
            # Extract r_PF
            pt_from_frame = points[:, i].reshape(2,1)
            
            # Get r_PO
            pt = pt_from_frame - frame_from_origin
            
            # Flip y-axis w.r.t. map height to match occupancy map
            # Occupancy map counts left and down
            # Robot coordinates count left and up
            pt[1] = h_m - pt[1]
            
            # Convert meters to pixels
            pt_occ_map = (pt / res_m_per_px).astype(int) # pixels must be in an integer grid
            
            # Input into grid
            cells[:, i] = pt_occ_map.squeeze()
        
        # Output cells (pixels) from meters
        return cells

    def points_to_robot_circle(self, points):
        # points: a series of (2xN) points of interest from the map reference to calculate disks for
        #       | [x1, x2, ..., xN] |
        # i.e.  | [y1, y2, ..., yN] |
        #
        # Get the cells corresponding to each (x,y) pair.
        # For each cell, construct a set of cells corresponding to the robot's area.
        # Each (x,y) pair generates an array listing each cell the robot enchroaches on.
        #
        # Output: list with a set of occupied cells for each (x,y) pair
        # | --------------------- |
        # | | [x1_1, x1_2, ...] | |
        # | | [y1_1, y1_2, ...] | |
        # | --------------------- |
        # | | [x2_1, x2_2, ...] | |
        # | | [y2_1, y2_2, ...] | |
        # | --------------------- |
        # |          ...          |
        
        # Output
        occ_cells = []
        
        # Extract map and robot properties
        res_m_per_px = self.map_settings_dict['resolution']
        radius_px = self.robot_radius / res_m_per_px
        
        # Get cells from points
        cells = self.point_to_cell(points)
        
        # Get occupied cells from each center cell
        num_cells = cells.shape[1]
        for i in range(num_cells):
            # Get robot center
            [x], [y] = cells[:, i].reshape(2, 1)
            
            # Get all occupied cells
            xs, ys = disk( (x, y), radius_px, shape=self.map_shape)
            
            # Add to output
            occ_cells.append(np.vstack([ xs, ys ]))
        
        # Return occupied cells
        return np.array(occ_cells)
    
    def point_collision_free(self, point):
        # IN PROGRESS
        #
        # If the (x,y) coordinate in the map is white: it's free.
        # If it's black: it's a wall
        return self.occupancy_map[
            point[1], # y-coordinate is column, indexed first
            point[0] # x-coordinate is row, indexed
        ]
    
    def trajectory_collision_free(self, traj):
        # IN PROGRESS
        #
        # Get occupied cells
        points = traj[:2, :]
        occupied_raw = self.points_to_robot_circle(points)
        
        # Unique coordinates
        # Eliminate (x,y) overlapping from several points in the trajectory
        occupied = np.unique(occupied_raw.transpose(0, 2, 1).reshape(-1,2), axis=1)
        for point in occupied:
            # Fail if any point is a collision
            if not self.point_collision_free(point):
                return False
        return True

    # Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    # RRT* specific functions
    def ball_radius(self):
        # Close neighbor distance
        card_V = len(self.nodes)
        return min(
            self.gamma_RRT * (np.log(card_V) / card_V) ** (1.0 / 2.0), self.epsilon
        )

    def connect_node_to_point(self, node_i, point_f):
        # Given two nodes find the non-holonomic path that connects them
        # Settings
        # node is a 3 by 1 node
        # point is a 2 by 1 point
        print(
            "TO DO: Implement a way to connect two already existing nodes (for rewiring)."
        )
        return np.zeros((3, self.num_substeps))

    def cost_to_come(self, trajectory_o):
        # The cost to get to a node from lavalle
        print("TO DO: Implement a cost to come metric")
        return 0

    def update_children(self, node_id):
        # Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    # Planner Functions
    def rrt_planning(self):
        # This function performs RRT on the given map and robot
        # You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        for i in range(
            1
        ):  # Most likely need more iterations than this to complete the map!
            # Sample map space
            point = self.sample_map_space()

            # Get the closest point
            closest_node_id = self.closest_node(point)

            # Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(
                self.nodes[closest_node_id].point, point
            )

            # Check for collisions
            print("TO DO: Check for collisions and add safe points to list of nodes.")

            # Check if goal has been reached
            print("TO DO: Check if at goal point.")
        return self.nodes

    def rrt_star_planning(self):
        # This function performs RRT* for the given map and robot
        for i in range(
            1
        ):  # Most likely need more iterations than this to complete the map!
            # Sample
            point = self.sample_map_space()

            # Closest Node
            closest_node_id = self.closest_node(point)

            # Simulate trajectory
            trajectory_o = self.simulate_trajectory(
                self.nodes[closest_node_id].point, point
            )

            # Check for Collision
            print("TO DO: Check for collision.")

            # Last node rewire
            print("TO DO: Last node rewiring")

            # Close node rewire
            print("TO DO: Near point rewiring")

            # Check for early end
            print("TO DO: Check for early end")
        return self.nodes

    def recover_path(self, node_id=-1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path


def main():
    # Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    # robot information
    goal_point = np.array([[10], [10]])  # m
    stopping_dist = 0.5  # m

    # RRT precursor
    path_planner = PathPlanner(
        map_filename, map_setings_filename, goal_point, stopping_dist
    )
    nodes = path_planner.rrt_star_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    # Leftover test functions
    np.save("shortest_path.npy", node_path_metric)


if __name__ == "__main__":
    main()
