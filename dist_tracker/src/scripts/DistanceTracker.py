#!/usr/bin/env python

# Distance Tracker
# Given a camera for input, will output the distance between the camera and 
#   a predetermined (hardcoded) target
#
# Author: Braden Cook
# Reference: pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

import rospy
import roslib
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge

class DistanceTracker:
    """ Distance Tracker - tracks the distance between a yellow block and a camera """
    def __init__(self):
        #taget object is yellow
        self.hsv_lower_lower = (14,55,55)
        self.hsv_lower_upper = (30,255,235)
        self.hsv_upper_lower = self.hsv_lower_lower
        self.hsv_upper_upper = self.hsv_lower_upper

        #focal length and real height parameters
        self.focal_length = 993.0
        self.real_height = 1.25

    def find_distance(self, focal_length, real_height, image):
        """ 
        Finds the distance from the camera to a targer of known height
        (in an upright orientation)

        params:
            focal_length (float): the focal length of the camera
            real_height (float): the real world height of the target object 
                 standing in an upright position
            image: the cv2 image from which to seach for the target object
        returns: 
            (float) the estimated distance between the target object and
            and the camera computed via triangle similarity
        """
        pixel_height = self.find_pixel_height(image)
        if pixel_height == 0:
            return 0
        return (focal_length * real_height) / pixel_height

    def find_pixel_height(self, image):
        """
        Searches for the target object and returns its height in pixels or 0
        if not found
        """
        #Blur the image, put into HSV color scale, and create an image mask 
        img_blur = cv2.GaussianBlur(image, (5,5), 0)
        img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        mask_l = cv2.inRange(img_hsv, self.hsv_lower_lower, self.hsv_lower_upper)
        mask_u = cv2.inRange(img_hsv, self.hsv_upper_lower, self.hsv_upper_upper)
        mask = cv2.bitwise_or(mask_l, mask_u)

        #find the largest contour of the mask or return 0 if target is not there
        img, cnts, cnt_hier = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            return 0        
        c = max(cnts, key=cv2.contourArea)

        #return the height of the target object using minAreaRect
        return cv2.minAreaRect(c)[1][1]

    def run(self, ros_data):
        """Run the Distance Tracker with the input from the camera"""
        bridge = CvBridge()
        cv_image = bridge.compressed_imgmsg_to_cv2(ros_data, desired_encoding='passthrough')
        dist = self.find_distance(self.focal_length, self.real_height, cv_image)
        print("EST DISTANCE: " + str(dist) + ' inches') #want to come up with a better way to output this
        

if __name__ == '__main__':
    rospy.init_node('dist_tracker', anonymous=True)
    tracker = DistanceTracker()
    rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, tracker.run)
    rospy.spin()
