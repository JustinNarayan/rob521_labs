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
import rob521_lab2.nodes.pygame_utils as pygame_utils

def normalize_angle(angle):
    return np.atan2( np.sin(angle), np.cos(angle) ) # now in [-np.pi, np.pi]

def vdist(v1, v2):
    return np.linalg.norm(v1.flatten() - v2.flatten())

def heading(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.atan2(y2-y1, x2-x1)

def load_map(filename):
    # Get the filepath
    full_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "rob521_lab2", "maps", filename)
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
        os.path.join(os.path.dirname(__file__), "..", "..", "rob521_lab2", "maps", filename)
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
    
    def add_child(self, child_id):
        self.children_ids.append(child_id)        
    
    def get_cost(self):
        return self.cost

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
        self.search_dist_around_node = 10 # search [-15, 15] around closest node to goal in (x,y)
        
        # For willow, trim the edges
        if map_filename == "willowgarageworld_05res.png":
            self.bounds = np.array([
                [0, 50],
                [-47, 13]
            ])

        # Robot information
        self.robot_radius = 0.22  # m
        self.vel_max = 0.4  # m/s (Feel free to change!)
        self.rot_vel_max = 1  # rad/s (Feel free to change!)
        self.min_dTheta_for_just_rotation = 0.6*np.pi
        self.min_dTheta_for_closest = 0.4*np.pi

        # Goal Parameters
        self.goal_point = goal_point  # m
        self.stopping_dist = stopping_dist  # m

        # Trajectory Simulation Parameters
        self.timestep = 1.0  # s
        self.num_substeps = 10

        # Planning storage
        self.nodes = {
            0: Node(np.array([0,0,0]), -1, 0)
        }
        self.next_id_to_assign = 1

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
            (self.map_shape[1] * 0.5, self.map_shape[0] * 0.5),
            map_filename,
            self.occupancy_map.shape,
            self.map_settings_dict,
            self.goal_point,
            self.stopping_dist,
        )
        return
    
    # Get the next id
    def get_new_id(self):
        # auto-increment ids
        id = self.next_id_to_assign
        self.next_id_to_assign += 1
        return id
    
    # Get node by id
    def get_node_by_id(self, id):
        # find node by id, nodes is a dictionary
        if id in self.nodes:
            return self.nodes[id]
        return None
    
    # Add node
    def add_node(self, pose, parent_id=None, cost_from_parent=0):
        new_id = self.get_new_id()
        parent_node = self.get_node_by_id(parent_id)
        parent_node.add_child(new_id)
        new_node = Node(
            pose,
            parent_id,
            parent_node.get_cost() + cost_from_parent
        )
        self.nodes[new_id] = new_node

    # Functions required for RRT
    def sample_map_space(self, bounds=None):
        # Return an [x,y] coordinate to drive the robot towards
        # self.map_shape is height, width i.e. y, x
        # Determine bounds
        xmin, xmax = self.bounds[0, :]
        ymin, ymax = self.bounds[1, :]
        
        if bounds is not None:
            # Get intersection of bounds within true map bounds
            xmin_spec, xmax_spec = bounds[0, :]
            ymin_spec, ymax_spec = bounds[1, :]
            xmin, ymin = max(xmin, xmin_spec), max(ymin, ymin_spec)
            xmax, ymax = min(xmax, xmax_spec), min(ymax, ymax_spec)
        
        x = np.random.randint(xmin, xmax)
        y = np.random.randint(ymin, ymax)
        return np.vstack((x, y))

    def check_if_duplicate(self, pose):
        # Check if any node has the same [x,y]
        tol_pos = 2e-4
        tol_theta = np.pi*0.01
        
        for id in self.nodes.keys():
            node_pose = self.nodes[id].get_pose()
            
            if (
                (vdist(node_pose[:2], np.array(pose[:2])) < tol_pos) and
                (abs(normalize_angle(pose[2] - node_pose[2])) < tol_theta)
            ):
                return True
        return False

    def closest_node(self, point, consider_heading=True):
        # finds closest node given an x,y point
        min_dist = float("inf")
        closest_id = 0
        # Iterate through nodes
        for id in self.nodes.keys():
            node_pose = self.nodes[id].pose
            node_xy = node_pose[:2]
            
            # Omit nodes that are not straight ahead
            theta_from_node = heading(node_xy, point)
            if abs(normalize_angle(theta_from_node - node_pose[2])) > self.min_dTheta_for_closest:
                continue
            
            # Calculate distance
            dist = vdist(node_xy, np.array(point))
            if dist < min_dist:
                min_dist = dist
                closest_id = id
        return closest_id
    
    def simulate_trajectory(self, node_i: Node, point_s):
        # A starting node and goal point is selected in meters.
        # 
        # Linear and rotation velocity commands are chosen for the robot in the global frame.
        # These commands, plus the starting node are passed to the robot.
        # A global output trajectory is produced:
        #   - x,y coordinates in the global frame
        #   - theta heading in the global frame
        #
        # No collision checking
        v, w = self.robot_controller(node_i, point_s)
        return self.trajectory_rollout(v, w, node_i.get_pose())

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
        v,w = 0,0
        if abs(dTheta) > self.min_dTheta_for_just_rotation:
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
        starting_pose
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
        t = np.linspace(0, self.timestep, self.num_substeps)
        
        # Initial pose
        x_i, y_i, theta_i = starting_pose.flatten()
        
        # Compute trajectory
        xs, ys = [], []
        thetas = w * t # same regardless of w
        # w = 0
        if w == 0:
            xs = v * t * np.cos(theta_i)
            ys = v * t * np.sin(theta_i)
        # w != 0
        else:
            xs = (v/w) * ( np.sin(theta_i + w*t) - np.sin(theta_i) )
            ys = (v/w) * (-np.cos(theta_i + w*t) + np.cos(theta_i) )
            
        # Account for starting pose
        xs = xs + x_i
        ys = ys + y_i
        thetas = normalize_angle(thetas+theta_i)
        
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
            ymax, xmax = self.map_shape
            xs, ys = disk( (x, y), radius_px, shape=(xmax, ymax))
            
            # Add to output
            if len(xs) == 0:
                continue
            occ_cells.append(np.vstack([ xs, ys ]))
        
        # Return occupied cells
        return occ_cells
    
    def cell_collision_free(self, cell):
        # If the (x,y) coordinate in the map is white: it's free.
        # If it's black: it's a wall
        # Cell dimensions not checked -- assume valid positions.
        # If this fails due to dimensions, it means cells have been mismapped elsewhere.
        return self.occupancy_map[
            cell[1], # y-coordinate is column, indexed first
            cell[0] # x-coordinate is row, indexed second
        ]
    
    def cells_collision_free(self, cells):
        # Check if a set of sets of cells is collision free
        # Each outer set is a "starting point".
        # Each inner set is the cells occupied if at that "starting point".
        # Cells is a list of length N or np arrays of shape (2, M)
        # N is the number of sets of cells.
        # M is the number of cells in a sets.
        num_cells_sets = len(cells)
        collisions = np.ones(num_cells_sets)
        
        for i in range(num_cells_sets): # Each set of cells
            these_cells = cells[i].reshape(2, -1)
            num_cells = these_cells.shape[1]
            # Iterate through cells and check if collision free
            for cell_i in range(num_cells):
                cell = these_cells[:, cell_i]
                collision_free = self.cell_collision_free(cell)
                
                if not collision_free:
                    collisions[i] = 0 # collision!
                    continue
        
        # For each set (of sets of cells), 0 = Collisions, 1 = No Collision
        return collisions

    # Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    # RRT* specific functions
    def ball_radius(self):
        # Close neighbor distance
        card_V = len(self.nodes)
        return min(
            self.gamma_RRT * (np.log(card_V) / card_V) ** (1.0 / 2.0), self.epsilon
        )

    def connect_node_to_point(self, node_i: Node, point_f):
        # Construct a trajectory from a starting pose to a final point. 
        # Final heading does not matter.
        # To guarantee arrival at the point, execute a turn maneuver to align the heading.
        # Then move straight.
        # Assume point is NOT a duplicate of node.
        
        # Extract information
        x, y, theta = node_i.get_pose()
        x_f, y_f = point_f
        sec_per_step = self.timestep / self.num_substeps
        vel_max_eff = self.vel_max * sec_per_step # scaled to timestamp
        rot_vel_max_eff = self.rot_vel_max * sec_per_step # scaled to timestamp
        
        # Rotation step
        target_heading = heading([x, y], [x_f, y_f])
        heading_error = normalize_angle(target_heading - theta)

        # Compute steps for rotational trajectory
        steps_to_rotate = max(int(np.ceil(abs(heading_error/rot_vel_max_eff))), 1) # guarantee one step
        
        # Create the required rotation steps
        traj_r = []
        for i in range(steps_to_rotate - 1):
            # Rotate
            theta += rot_vel_max_eff * np.sign(heading_error)
            # Required rotation steps up to, but not including, final heading
            traj_r.append([x, y, theta])
        
        # Append final step
        theta = target_heading
        traj_r.append([x, y, theta])
        
        # Format as 3xN, with x, y, theta as rows
        traj_r = np.array(traj_r).T
        
        # Translation step
        dist = np.hypot(x_f-x, y_f-y)
        
        # Compute steps for linear trajectory
        steps_to_translate = max(int(np.ceil(dist / vel_max_eff)), 1) # guarantee one 
        
        # Create the required translation steps
        traj_t = []
        for i in range(steps_to_translate - 1):
            # Move
            x += vel_max_eff * np.cos(theta)
            y += vel_max_eff * np.sin(theta)
            # Required translations steps up to, but not including, final position
            traj_t.append([x, y, theta])
        
        # Append final step
        traj_t.append([x_f, y_f, target_heading])
        
        # Format as 3xN, with x, y, theta as rows
        traj_t = np.array(traj_t).T
        
        # Append trajectories
        return np.hstack([traj_r, traj_t])

    def cost_to_come(self, traj):
        # Segment lengths
        lengths = vdist(traj[:2, 1:], traj[:2, :-1])
        
        # Orientation change
        theta_diffs = normalize_angle(np.diff(traj[2, :]))
        
        # Approximate control "effort"
        dt = self.timestep / self.num_substeps
        v = lengths / dt
        w = theta_diffs / dt
        
        # Costs (tunable)
        K_lengths = 1
        K_theta = 0.1
        K_control = 0.01 * dt
        
        # Compute cost
        cost = np.sum(
            K_lengths*lengths + \
            K_theta*theta_diffs + \
            K_control*(v**2 + w**2)*dt
        )
        
        return cost

    def update_children(self, node_id):
        # Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    # Planner Functions
    def rrt_planning(self):        
        # RRT
        max_iterations = int(1e8)
        tol = self.stopping_dist # m
        
        for i in range(
            max_iterations
        ):
            # Draw pygame
            for event in pygame.event.get():
                pass
                # if event.type == pygame.QUIT:
                #     pygame.quit()
                #     return self.nodes  # exit planning safely
            
            # Debug message
            if (i % 500 == 0) and i>0:
                print(f'iteration: {i}')
                
            # Sample map space
            # Sample points around the closest node to the goal
            closest_node_id_to_goal = self.closest_node(self.goal_point, consider_heading=False)
            near_x, near_y = self.nodes[closest_node_id_to_goal].get_pose()[:2]
            bounds_to_search = np.array([
                [near_x - self.search_dist_around_node, near_x + self.search_dist_around_node],
                [near_y - self.search_dist_around_node, near_y + self.search_dist_around_node],
            ])
            
            # Every few searches check the goal node
            if (i % 20 == 0):
                bounds_to_search = np.array([
                    [self.goal_point[0] - 0.5, self.goal_point[0] + 0.5],
                    [self.goal_point[1] - 0.5, self.goal_point[1] + 0.5]
                ])
            
            point = self.sample_map_space(bounds=bounds_to_search)

            # Get the closest nod
            closest_node_id = self.closest_node(point)

            # Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(
                self.nodes[closest_node_id], point
            )
            
            # Pose is the end of the trajectory
            final_pose = trajectory_o[:, -1]
            
            # Do not bother if duplicate
            if self.check_if_duplicate(final_pose):
                continue
            
            # Get cells occupied by trajectory
            occ_cells = self.points_to_robot_circle(trajectory_o[:2, :]) # omit theta
            
            # Check if cells colliding
            cells_collision_free = self.cells_collision_free(occ_cells)
            collision = np.any(cells_collision_free == 0)
            
            # Add node if safe
            if not collision:
                # Parent ID is the closest node
                parent_id = closest_node_id
                parent_pose = self.nodes[closest_node_id].get_pose()
                # Add node
                self.add_node(
                    final_pose, 
                    parent_id, 
                    cost_from_parent=self.cost_to_come(trajectory_o)
                )
                
                # PyGame drawing
                xf, yf = final_pose[:2]
                self.window.add_point(
                    [xf,yf],
                    radius=1.5,
                    color=pygame_utils.COLORS['b']
                )
                xp, yp = parent_pose[:2]
                
                self.window.add_line(
                    [xp, yp],
                    [xf, yf],
                    width=1,
                    color=pygame_utils.COLORS['g']
                )
                
                # Check if we are at the goal
                dist_to_goal = vdist(final_pose[:2], self.goal_point)
                if dist_to_goal < tol:
                    break
        self.window.save("success.png")
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
