#!/usr/bin/env python3
from __future__ import division, print_function
import time

import numpy as np
import rospy
import tf_conversions
import tf2_ros
import rosbag
import rospkg

# msgs
from turtlebot3_msgs.msg import SensorState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist, TransformStamped, Transform, Quaternion
from std_msgs.msg import Empty

from utils import convert_pose_to_tf, euler_from_ros_quat, ros_quat_from_euler


ENC_TICKS = 4096
RAD_PER_TICK = 0.001533981
WHEEL_RADIUS = .066 / 2
BASELINE = .287 / 2

'''
VALIDATION

Terminal 1 >> `roscore`
Terminal 2 >> `rosrun rob521_lab3 l3_estimate_robot_motion.py`
Terminal 3 >> `rosbag play <path to lab 3>/rosbags/loop.py`

Output (Terminal 2):
...
Wheel Odom: x: 0.000, y: 0.000, t: 3.142
Turtlebot3 Odom: x: 0.000, y: 0.002, t: 3.137
...
Wheel Odom: x: -0.048, y: -0.030, t: -2.783
Turtlebot3 Odom: x: 0.005, y: -0.039, t: -3.120

COMPARISON:
Expected init state (x,y,θ): [ 0.000 m, 0.000 m, 3.142 rad]
Observed init state (x,y,θ): [ 0.000 m, 0.002 m, 3.137 rad]

Expected final state (x,y,θ): [ 0.005 m, -0.039 m, -3.120 rad]
Observed final state (x,y,θ): [-0.048 m, -0.030 m, -2.783 rad]
'''

class WheelOdom:
    def __init__(self):
        # publishers, subscribers, tf broadcaster
        self.sensor_state_sub = rospy.Subscriber('/sensor_state', SensorState, self.sensor_state_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_cb, queue_size=1)
        self.wheel_odom_pub = rospy.Publisher('/wheel_odom', Odometry, queue_size=1)
        self.tf_br = tf2_ros.TransformBroadcaster()

        # attributes
        self.odom = Odometry()
        self.odom.pose.pose.position.x = 1e10
        self.wheel_odom = Odometry()
        self.wheel_odom.header.frame_id = 'odom'
        self.wheel_odom.child_frame_id = 'wo_base_link'
        self.wheel_odom_tf = TransformStamped()
        self.wheel_odom_tf.header.frame_id = 'odom'
        self.wheel_odom_tf.child_frame_id = 'wo_base_link'
        self.pose = Pose()
        # ------------ INSERT OUR CODE ------------
        # The ROS bag we're simulating starts at an Euler z of 3.137 (~ pi)
        # Why on Earth doesn't the starter code also make us start at the same orientation?
        # What a dumb oversight, that sets us up to fail.
        self.pose.orientation = ros_quat_from_euler(
            np.array([0, 0, np.pi])
        )
        # ------------------ DONE -----------------
        self.twist = Twist()
        self.last_enc_l = None
        self.last_enc_r = None
        self.last_time = None

        # rosbag
        rospack = rospkg.RosPack()
        path = rospack.get_path("rob521_lab3")
        self.bag = rosbag.Bag(path+"/motion_estimate.bag", 'w')

        # reset current odometry to allow comparison with this node
        reset_pub = rospy.Publisher('/reset', Empty, queue_size=1, latch=True)
        reset_pub.publish(Empty())
        while not rospy.is_shutdown() and (self.odom.pose.pose.position.x >= 1e-3 or self.odom.pose.pose.position.y >= 1e-3 or
               self.odom.pose.pose.orientation.z >= 1e-2):
            time.sleep(0.2)  # allow reset_pub to be ready to publish
        print('Robot odometry reset.')

        rospy.spin()
        self.bag.close()
        print("saving bag")

    def sensor_state_cb(self, sensor_state_msg):
        # Callback for whenever a new encoder message is published
        # set initial encoder pose
        if self.last_enc_l is None:
            self.last_enc_l = sensor_state_msg.left_encoder
            self.last_enc_r = sensor_state_msg.right_encoder
            self.last_time = sensor_state_msg.header.stamp
        else:
            # update calculated pose and twist with new data
            le = sensor_state_msg.left_encoder 
            re = sensor_state_msg.right_encoder
            time = sensor_state_msg.header.stamp

            # ------------ INSERT OUR CODE ------------
            
            # Extract yaw (euler angle z)
            euler = euler_from_ros_quat(self.pose.orientation)
            
            # Determine trigonometric values based on assumed (i.e. approx) theta at start of update
            cos_theta, sin_theta = np.cos(euler[2]), np.sin(euler[2])
            
            # Determine dx, dy, dt (dtheta)
            # θ used is assumed (i.e. approx) theta
            # r used is wheel radius (assumed to be equal)
            # b used is baseline
            # le, re should be in radians
            #
            #  ----     ---------   ----------------   ----
            # | dx |   | cos_θ 0 | |  (r/2)   (r/2) | | re |
            # | dy | = | sin_θ 0 | | (r/2b) -(r/2b) | | le |
            # | dθ |   |   0   1 |  ----------------   ----
            #  ---      ---------
            
            # Convert encoder readings to radians
            del_le_rad, del_re_rad = \
                (le - self.last_enc_l)*RAD_PER_TICK, (re - self.last_enc_r)*RAD_PER_TICK
            
            # Compute pose changes
            dx = cos_theta * ( (WHEEL_RADIUS/2)*(del_re_rad + del_le_rad) )
            dy = sin_theta * ( (WHEEL_RADIUS/2)*(del_re_rad + del_le_rad) )
            dtheta = ( (WHEEL_RADIUS/(2*BASELINE))*del_re_rad ) \
                - ( (WHEEL_RADIUS/(2*BASELINE))*del_le_rad )
                
            # Differentiate pose changes by time
            dt = max( (time - self.last_time).to_sec() , 1e-10) # will be > 0
            x_dot, y_dot, theta_dot = dx/dt, dy/dt, dtheta/dt

            # Update odometry message
            self.pose.position.x += dx
            self.pose.position.y += dy
            self.pose.orientation = ros_quat_from_euler(
                np.array([euler[0], euler[1], euler[2] + dtheta]) # add yaw to initial euler angle and convert to quaternion
            )
            self.twist.linear.x = x_dot
            self.twist.linear.y = y_dot
            self.twist.angular.z = theta_dot
            
            # Update most recent encoder values
            self.last_enc_l = le
            self.last_enc_r = re
            self.last_time = time
            
            # ------------------ DONE -----------------

            # publish the updates as a topic and in the tf tree
            current_time = rospy.Time.now()
            self.wheel_odom_tf.header.stamp = current_time
            self.wheel_odom_tf.transform = convert_pose_to_tf(self.pose)
            self.tf_br.sendTransform(self.wheel_odom_tf)

            self.wheel_odom.header.stamp = current_time
            self.wheel_odom.pose.pose = self.pose
            self.wheel_odom.twist.twist = self.twist
            self.wheel_odom_pub.publish(self.wheel_odom)

            self.bag.write('odom_est', self.wheel_odom)

            # for testing against actual odom
            print("Wheel Odom: x: %2.3f, y: %2.3f, t: %2.3f" % (
                self.pose.position.x, self.pose.position.y, euler_from_ros_quat(self.pose.orientation)[2]
            ))
            print("Turtlebot3 Odom: x: %2.3f, y: %2.3f, t: %2.3f" % (
                self.odom.pose.pose.position.x, self.odom.pose.pose.position.y,
                euler_from_ros_quat(self.odom.pose.pose.orientation)[2]
            ))

    def odom_cb(self, odom_msg):
        # get odom from turtlebot3 packages
        self.odom = odom_msg
        self.bag.write('odom_onboard', self.odom)

    def plot(self, bag):
        data = {"odom_est":{"time":[], "data":[]}, 
                "odom_onboard":{'time':[], "data":[]}}
        for topic, msg, t in bag.read_messages(topics=['odom_est', 'odom_onboard']):
            print(msg)


if __name__ == '__main__':
    try:
        rospy.init_node('wheel_odometry')
        wheel_odom = WheelOdom()
    except rospy.ROSInterruptException:
        pass