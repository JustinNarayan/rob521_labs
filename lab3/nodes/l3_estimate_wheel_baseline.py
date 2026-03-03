#!/usr/bin/env python3

import rospy
import numpy as np
import threading
from turtlebot3_msgs.msg import SensorState
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist

INT32_MAX = 2**31
NUM_ROTATIONS = 3 
TICKS_PER_ROTATION = 4096
WHEEL_RADIUS = 0.066 / 2 #In meters


class wheelBaselineEstimator():
    def __init__(self):
        rospy.init_node('encoder_data', anonymous=True) # Initialize node

        #Subscriber bank
        rospy.Subscriber("cmd_vel", Twist, self.startStopCallback)
        rospy.Subscriber("sensor_state", SensorState, self.sensorCallback) #Subscribe to the sensor state msg

        #Publisher bank
        self.reset_pub = rospy.Publisher('reset', Empty, queue_size=1)

        #Initialize variables
        self.left_encoder_prev = None
        self.right_encoder_prev = None
        self.del_left_encoder = 0
        self.del_right_encoder = 0
        self.isMoving = False #Moving or not moving
        self.lock = threading.Lock()

        #Reset the robot 
        reset_msg = Empty()
        self.reset_pub.publish(reset_msg)
        print('Ready to start wheel radius calibration!')
        return

    def safeDelPhi(self, a, b):
        #Need to check if the encoder storage variable has overflowed
        diff = np.int64(b) - np.int64(a)
        if diff < -np.int64(INT32_MAX): #Overflowed
            delPhi = (INT32_MAX - 1 - a) + (INT32_MAX + b) + 1
        elif diff > np.int64(INT32_MAX) - 1: #Underflowed
            delPhi = (INT32_MAX + a) + (INT32_MAX - 1 - b) + 1
        else:
            delPhi = b - a  
        return delPhi

    def sensorCallback(self, msg):
        #Retrieve the encoder data form the sensor state msg
        self.lock.acquire()
        if (self.left_encoder_prev is None) or (self.right_encoder_prev is None): 
            self.left_encoder_prev = msg.left_encoder #int32
            self.right_encoder_prev = msg.right_encoder #int32
        else:
            #Calculate and integrate the change in encoder value
            self.del_left_encoder += self.safeDelPhi(self.left_encoder_prev, msg.left_encoder)
            self.del_right_encoder += self.safeDelPhi(self.right_encoder_prev, msg.right_encoder)

            #Store the new encoder values
            self.left_encoder_prev = msg.left_encoder #int32
            self.right_encoder_prev = msg.right_encoder #int32
        self.lock.release()
        return

    def startStopCallback(self, msg):
        if self.isMoving is False and np.absolute(msg.angular.z) > 0:
            self.isMoving = True #Set state to moving
            print('Starting Calibration Procedure')

        elif self.isMoving is True and np.isclose(msg.angular.z, 0):
            self.isMoving = False #Set the state to stopped
            
            self.lock.acquire() # ought to have lock when reading encoders

            # ------------ INSERT OUR CODE ------------
            
            # Goal is to compute the wheel separation / baseline
            # The robot has just performed three rotations
            # i.e. the wheels have swept an angular distance of NUM_ROTATIONS*(2*pi)
            # The total circumference swept by each wheel will be:
            #    C = (NUM_ROTATIONS) * pi * baseline = [3 * (2 * pi * radius)]
            # Each wheel's encoder values will return the traveled distance
            # i.e. the wheels sweep out C
            # Both encoders will differ due to error, so it may be best to take their average
            # Each full rotation of the wheel is (2*pi)*WHEEL_RADIUS
            # Each full rotation of the wheel is also TICKS_PER_ROTATION from the encoder
            # The distance swept by each wheel is thus:
            #    d = {(# encoder ticks) / TICKS_PER_ROTATION} * \    num rotations *
            #       {(2*pi)*WHEEL_RADIUS}                            distance per rotation
            # Taking the average (in theory they should be equal) of both wheel's d values:
            #    baseline = (0.5 * (d_l + d_r)) / (NUM_ROTATIONS * pi)
            
            d_l = abs(self.del_left_encoder / TICKS_PER_ROTATION) * (2*np.pi*WHEEL_RADIUS)
            d_r = abs(self.del_right_encoder / TICKS_PER_ROTATION) * (2*np.pi*WHEEL_RADIUS)
            d_avg = 0.5*(d_l + d_r)
            
            separation = d_avg / (NUM_ROTATIONS *  np.pi)
            
            # ------------------ DONE -----------------
            print('Calibrated Separation: {} m'.format(separation))

            #Reset the robot and calibration routine
            self.left_encoder_prev = None
            self.right_encoder_prev = None
            self.del_left_encoder = 0
            self.del_right_encoder = 0
            self.lock.release()
            reset_msg = Empty()
            self.reset_pub.publish(reset_msg)
            print('Resetted the robot to calibrate again!')

        return


if __name__ == '__main__':
    Estimator = wheelBaselineEstimator() #create instance
    rospy.spin()
