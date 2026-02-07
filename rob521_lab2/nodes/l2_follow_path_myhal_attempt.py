#!/usr/bin/env python3
from __future__ import division, print_function
import os

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cityblock
import rospy
from skimage.draw import disk
import tf2_ros

# msgs
from geometry_msgs.msg import TransformStamped, Twist, PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker
import yaml

# ros and se2 conversion utils
import utils

# Goal Tolerances
TRANS_GOAL_TOL = 0.5  # m, tolerance to consider a goal complete
ROT_GOAL_TOL = 0.5  # rad, tolerance to consider a goal complete

# Options for Velocities
TRANS_VEL_OPTS = [0, 0.01, 0.1]  # m/s, max of real robot is .26
ROT_VEL_OPTS = np.linspace(-0.8, 0.8, 9)  # rad/s, max of real robot is 1.82

# Control frequency
CONTROL_RATE = 8  # Hz, how frequently control signals are sent

# Time horizon simulation
CONTROL_HORIZON = 5  # seconds. if this is set too high and INTEGRATION_DT is too low, code will take a long time to run!
INTEGRATION_DT = 0.025  # s, delta t to propagate trajectories forward by

# Collision Checks
COLLISION_RADIUS = 0.225  # m, radius from base_link to use for collisions, min of 0.2077 based on dimensions of .281 x .306
HEURISTIC_RADII = [0.25, 0.275, 0.3, 0.325]
HEURISTIC_RADII_INFINITE = 0.35 # this radii suggests robot is "infinitely far" from obstacles for purpose of cost. Ideal

# Costs
COST_LIN_DIST = 1 # per "m" for [0, inf] -> [good, bad]. 0 heuristic means at goal. inf heuristic means very far from goal.
COST_ROT_DIST = 5 # per "rad" for [0, pi] -> [good, bad]. 0 heuristic means aligned with goal. pi heuristic means opposite from goal.
COST_OBS_DIST = 0 # per "m" for [0, 1] -> [good, bad]. 0 heuristic means > 0.325 m away from obstacles. 0.1 means <= 0.25 m away from obstacles
DIST_TO_CHECK_ROT = 0.3 # m


def normalize_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi # now in [-np.pi, np.pi]

def vdist(v1, v2):
    return np.linalg.norm(v1.flatten() - v2.flatten())

#Map Handling Functions
def load_map(filename):
    import matplotlib.image as mpimg
    import cv2 
    im = cv2.imread("../maps/" + filename)
    im = cv2.flip(im, 0)
    # im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    im_np = np.logical_not(im_np)     #for ros
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

class PathFollower():
    def __init__(self):
        # time full path
        self.path_follow_start_time = rospy.Time.now()

        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)  # time to get buffer running

        # constant transforms
        self.map_odom_tf = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(2.0)).transform
        print(self.map_odom_tf)

        # subscribers and publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.global_path_pub = rospy.Publisher('~global_path', Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher('~local_path', Path, queue_size=1)
        self.collision_marker_pub = rospy.Publisher('~collision_marker', Marker, queue_size=1)

        # map
        # map = rospy.wait_for_message('/map', OccupancyGrid)
        # self.map_np = np.array(map.data).reshape(map.info.height, map.info.width)
        # self.map_resolution = round(map.info.resolution, 5)
        # self.map_origin = -utils.se2_pose_from_pose(map.info.origin)  # negative because of weird way origin is stored
        # self.map_nonzero_idxes = np.argwhere(self.map_np)
        map_filename = "myhal.png"
        occupancy_map = load_map(map_filename)
        self.map_shape = occupancy_map.shape
        self.map_settings_dict = load_map_yaml("myhal.yaml")
        self.map_np = occupancy_map
        self.map_resolution = 0.05
        self.map_origin = np.array([ 0.2 , 0.2 ,-0. ])
        self.map_nonzero_idxes = np.argwhere(self.map_np)
        
        # collisions
        self.collision_radius_pix = COLLISION_RADIUS / self.map_resolution
        self.collision_marker = Marker()
        self.collision_marker.header.frame_id = '/map'
        self.collision_marker.ns = '/collision_radius'
        self.collision_marker.id = 0
        self.collision_marker.type = Marker.CYLINDER
        self.collision_marker.action = Marker.ADD
        self.collision_marker.scale.x = COLLISION_RADIUS * 2
        self.collision_marker.scale.y = COLLISION_RADIUS * 2
        self.collision_marker.scale.z = 1.0
        self.collision_marker.color.g = 1.0
        self.collision_marker.color.a = 0.5

        # transforms
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_footprint', rospy.Time(0), rospy.Duration(2.0))
        self.pose_in_map_np = np.zeros(3)
        self.pos_in_map_pix = np.zeros(2)
        self.update_pose()

        # path variables
        cur_dir = os.path.dirname(os.path.realpath(__file__))

        # to use the temp hardcoded paths above, switch the comment on the following two lines
        self.path_tuples = np.load(os.path.join(cur_dir, 'shortest_path_rrt_heuristic.npy')).T

        self.path = utils.se2_pose_list_to_path(self.path_tuples, 'map')
        self.global_path_pub.publish(self.path)

        # goal
        self.cur_goal = np.array(self.path_tuples[0])
        self.cur_path_index = 0

        # trajectory rollout tools
        # self.all_opts is a Nx2 array with all N possible combinations of the t and v vels, scaled by integration dt
        # self.all_opts = np.array(np.meshgrid(TRANS_VEL_OPTS, ROT_VEL_OPTS)).T.reshape(-1, 2)
        
        self.all_opts = np.vstack([
            np.array([(v, 0) for v in TRANS_VEL_OPTS]),
            np.array([(0, w) for w in ROT_VEL_OPTS])
        ])

        # if there is a [0, 0] option, remove it
        all_zeros_index = (np.abs(self.all_opts) < [0.001, 0.001]).all(axis=1).nonzero()[0]
        if all_zeros_index.size > 0:
            self.all_opts = np.delete(self.all_opts, all_zeros_index, axis=0)
        self.all_opts_scaled = self.all_opts * INTEGRATION_DT

        self.num_opts = self.all_opts_scaled.shape[0]
        self.horizon_timesteps = int(np.ceil(CONTROL_HORIZON / INTEGRATION_DT))

        self.rate = rospy.Rate(CONTROL_RATE)

        rospy.on_shutdown(self.stop_robot_on_shutdown)
        self.follow_path()

    def follow_path(self):        
        while not rospy.is_shutdown():
            # timing for debugging...loop time should be less than 1/CONTROL_RATE
            tic = rospy.Time.now()
            self.update_pose()
            self.check_and_update_goal()

            # start trajectory rollout algorithm
            local_paths = np.zeros([self.horizon_timesteps + 1, self.num_opts, 3])
            local_paths[0] = np.atleast_2d(self.pose_in_map_np).repeat(
                self.num_opts, axis=0
            )

            ### SIMULATE TRAJECTORY
            ### Propogate trajectory, assuming perfect control of velocity and no dynamic effects
            # Extract current position
            _x, _y, _theta = self.pose_in_map_np
            # Iterate through all timesteps
            for t in range(1, self.horizon_timesteps + 1):
                # Iterate through all control options
                for o in range(1, self.num_opts):
                    opt = self.all_opts_scaled[o, :].flatten()
                    # Extract v, w
                    v, w = opt
                    # Extract previous x, y, theta
                    x, y, theta = None, None, None
                    if t == 1:
                        x, y, theta = _x, _y, _theta
                    else:
                        x, y, theta = local_paths[t-1, o, :].flatten()
                    # Step x, y, theta
                    if w == 0:
                        x += v*np.cos(theta)
                        y += v*np.sin(theta)
                    else:
                        x += (v/w) * ( np.sin(theta + w) - np.sin(theta))
                        y += (v/w) * (-np.cos(theta + w) + np.cos(theta))
                    theta += w
                    # Put into path
                    local_paths[t, o, :] = np.array([x,y,theta]).reshape(3,)
                pass

            # Check all trajectory points for collisions
            # First find the closest collision point in the map to each local path point
            local_paths_pixels = (
                self.map_origin[:2] + local_paths[:, :, :2]
            ) / self.map_resolution
            local_paths_lowest_collision_dist = np.ones(self.num_opts) * 50

            ### Check paths for collisions
            cells = []
            for o in range(self.num_opts):
                cells.append(local_paths_pixels[:, o, :2].T)
            collisions, sets_colliding = self.cells_collision_free(cells)

            # Remove colliding trajectories
            valid_opts = np.delete(self.all_opts, sets_colliding, axis=0)
            paths = np.delete(local_paths, sets_colliding, axis=1)
            paths_pixels = np.delete(local_paths_pixels, sets_colliding, axis=1)
            
            ## Calculate heuristics of trajectories
            # Linear and rotational distance from goal
            curr_dist_from_goal = vdist(self.pose_in_map_np[:2], self.cur_goal[:2])
            final_poses = paths[-1, :, :]
            lin_dists_to_goal, rot_dists_from_goal = [], []
            for o in range(final_poses.shape[0]):
                lin_dists_to_goal.append(vdist(final_poses[o, :2], self.cur_goal[:2]))
                if curr_dist_from_goal <= DIST_TO_CHECK_ROT:
                    rot_dists_from_goal.append(np.abs(normalize_angle(final_poses[o,2] - self.cur_goal[2])))
                else:
                    rot_dists_from_goal.append(0)
            # Closeness to object
            dists_from_obstacles = [HEURISTIC_RADII_INFINITE for o in valid_opts]
            for o in range(len(valid_opts)):
                point = paths_pixels[-1, o, :2]
                # Check increasing radii of robot at end of trajectory to assess closeness to obstacles
                for radii in HEURISTIC_RADII:
                    cells = self.points_to_robot_circle(np.array([point]).T)
                    _, sets_colliding = self.cells_collision_free(cells)
                    
                    # If collided, store this radii
                    if len(sets_colliding) > 0:
                        dists_from_obstacles[o] = radii
                        break
                # If all checked radii were free, robot is far from obstacles.
            # Scale dists from obstacles so 0 = "infinitely far"
            adjusted_dists_from_obstacles = [dist-HEURISTIC_RADII_INFINITE for dist in dists_from_obstacles]
            
            # Calculate costs
            lin_cost = np.array([COST_LIN_DIST * dist for dist in lin_dists_to_goal]).flatten()
            rot_cost = np.array([COST_ROT_DIST * dist for dist in rot_dists_from_goal]).flatten()
            obs_cost = np.array([COST_OBS_DIST * dist for dist in adjusted_dists_from_obstacles]).flatten()
            final_cost = (lin_cost + rot_cost + obs_cost).flatten() 
            
            # Choose best cost
            if final_cost.size == 0:  # hardcoded recovery if all options have collision
                control = [-0.1, 0] # go back
            else:
                best_opt = final_cost.argmin()
                control = valid_opts[best_opt]
                
                # Publish pose list as path
                self.local_path_pub.publish(
                    utils.se2_pose_list_to_path(paths[:, best_opt], "map")
                )

            # Send command
            self.cmd_pub.publish(utils.unicyle_vel_to_twist(control))

            # uncomment out for debugging if necessary
            print("Selected control: {control}, Loop time: {time}, Max time: {max_time}".format(
                control=control, time=(rospy.Time.now() - tic).to_sec(), max_time=1/CONTROL_RATE))

            self.rate.sleep()

    def update_pose(self):
        # Update numpy poses with current pose using the tf_buffer
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_footprint', rospy.Time(0)).transform
        self.pose_in_map_np[:] = [self.map_baselink_tf.translation.x, self.map_baselink_tf.translation.y,
                                  utils.euler_from_ros_quat(self.map_baselink_tf.rotation)[2]]
        self.pos_in_map_pix = (self.map_origin[:2] + self.pose_in_map_np[:2]) / self.map_resolution
        self.collision_marker.header.stamp = rospy.Time.now()
        self.collision_marker.pose = utils.pose_from_se2_pose(self.pose_in_map_np)
        self.collision_marker_pub.publish(self.collision_marker)

    def check_and_update_goal(self):
        # iterate the goal if necessary
        dist_from_goal = vdist(self.pose_in_map_np[:2], self.cur_goal[:2])
        rot_dist_from_goal = np.abs(normalize_angle(self.pose_in_map_np[2] - self.cur_goal[2]))
        
        num_goals = len(self.path_tuples[self.cur_path_index])
        trans_goal_tol_eff = TRANS_GOAL_TOL
        # if self.cur_path_index >= num_goals-1:
        #     trans_goal_tol_eff = TRANS_GOAL_TOL/5
        
        # abs_angle_diff = np.abs(self.pose_in_map_np[2] - self.cur_goal[2]) # old implementation
        # rot_dist_from_goal = min(np.pi * 2 - abs_angle_diff, abs_angle_diff)
        if dist_from_goal < trans_goal_tol_eff:# and rot_dist_from_goal < ROT_GOAL_TOL:
            rospy.loginfo(
                "Goal {goal} at {pose} complete.".format(
                    goal=self.cur_path_index, pose=self.cur_goal
                )
            )
            if self.cur_path_index == len(self.path_tuples) - 1:
                rospy.loginfo(
                    "Full path complete in {time}s! Path Follower node shutting down.".format(
                        time=(rospy.Time.now() - self.path_follow_start_time).to_sec()
                    )
                )
                rospy.signal_shutdown(
                    "Full path complete! Path Follower node shutting down."
                )
            else:
                self.cur_path_index += 1
                self.cur_goal = np.array(self.path_tuples[self.cur_path_index])
        else:
            rospy.logdebug(
                "Goal {goal} at {pose}, trans error: {t_err}, rot error: {r_err}.".format(
                    goal=self.cur_path_index,
                    pose=self.cur_goal,
                    t_err=dist_from_goal,
                    r_err=rot_dist_from_goal,
                )
            )

    def stop_robot_on_shutdown(self):
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Published zero vel on shutdown.")

    '''
    
    FUNCTIONS COPIED FROM l2_planning.py
    
    Why on Earth would the teaching team not create these files in any remotely logical way so we could easily just import PathPlanner? All of the initialization code in this file and l2_planning.py with map files are completely contradictory. This file layout is truly absurd and moronic. Do better. If they can't be bothered to care, why should students?
    
    '''
    
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
        res_m_per_px = self.map_resolution
        w_px, h_px = self.map_np.shape
        w_m, h_m = w_px * res_m_per_px, h_px * res_m_per_px
        
        # "Origin" property is offset of map reference frame from the true origin
        # Provided r_PF (point from frame), get r_PO (point from origin)
        # r_PO = r_PF - r_FO (frame from origin)
        frame_from_origin = np.array(self.map_origin[:2]).reshape(2,1)
        
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

    def points_to_robot_circle(self, points, radius=COLLISION_RADIUS):
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
        res_m_per_px = self.map_resolution
        radius_px = radius / res_m_per_px
        
        # Get cells from points
        cells = self.point_to_cell(points)
        
        # Get occupied cells from each center cell
        num_cells = cells.shape[1]
        for i in range(num_cells):
            # Get robot center
            [x], [y] = cells[:, i].reshape(2, 1)
            
            # Get all occupied cells
            ymax, xmax = self.map_np.shape
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
        # The occupancy map appears extremely finicky for Myhal
        # Use the hardcoded obstacle locations in meters instead
        # Locations are in the map-frame (i.e. 0 is the "top")
        
        # Map dimensions
        res = self.map_settings_dict["resolution"]
        h, w = np.array(self.map_shape) * res
        o_x, o_y = np.array(self.map_settings_dict["origin"][:2])
        
        # Get position in meters in robot frame
        robot_frame_x = cell[0] * res
        robot_frame_y = cell[1] * res
        
        # Augment to map origin
        x = robot_frame_x - o_x
        y = robot_frame_y - o_y
        
        # Check if in bounds
        # dP = COLLISION_RADIUS
        # if ( (x<-dP) or (x>w+dP) or (y<-dP) or (y>h+dP) ):
        #     return False # wall
        
        # Check if in obstacle
        # for obs in self.map_settings_dict["obstacles"].values():
        #     # Extract dimensions
        #     obs_x, obs_y, obs_w, obs_h, _ = obs
        #     x_l, x_r = obs_x, obs_x + obs_w
        #     y_t, y_b = obs_y, obs_y + obs_h
            
        #     # Determine collision
        #     if ( (x>x_l) and (x<x_r) ) and ( (y>y_t) and (y<y_b) ):
        #         return False
            
        # No collision
        return True
    
    def cells_collision_free(self, cells):
        # Check if a set of sets of cells is collision free
        # Each outer set is a "starting point".
        # Each inner set is the cells occupied if at that "starting point".
        # Cells is a list of length N or np arrays of shape (2, M)
        # N is the number of sets of cells.
        # M is the number of cells in a sets.
        num_cells_sets = len(cells)
        collisions = np.ones(num_cells_sets)
        sets_colliding = []
        
        for i in range(num_cells_sets): # Each set of cells
            these_cells = cells[i].reshape(2, -1)
            num_cells = these_cells.shape[1]
            # Iterate through cells and check if collision free
            for cell_i in range(num_cells):
                cell = these_cells[:, cell_i]
                collision_free = self.cell_collision_free(cell)
                
                if not collision_free:
                    collisions[i] = 0 # collision!
                    # Add to free sets
                    sets_colliding.append(i)
                    continue
        
        # For each set (of sets of cells), 0 = Collisions, 1 = No Collision
        return collisions, sets_colliding


if __name__ == "__main__":
    try:
        rospy.init_node('path_follower', log_level=rospy.DEBUG)
        pf = PathFollower()
    except rospy.ROSInterruptException:
        pass