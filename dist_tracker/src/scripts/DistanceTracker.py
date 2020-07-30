#!/usr/bin/env python

# Distance Tracker
# Given a camera for input, will output the distance between the camera and 
#   a predetermined (hardcoded) target
#
# Author: Braden Cook
# Reference: pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

import sys
import rospy
import roslib
from std_msgs.msg import String, Float64
from sensor_msgs.msg import Image, CompressedImage
import cv2
from cv_bridge import CvBridge

class DistanceTracker:
    """ 
    Distance Tracker - tracks the distance between a yellow block and a camera
    using triangle similarity 
    """
    def __init__(self):
        #taget object is yellow
        self.hsv_lower_lower = (14,55,55)
        self.hsv_lower_upper = (30,255,235)
        self.hsv_upper_lower = self.hsv_lower_lower
        self.hsv_upper_upper = self.hsv_lower_upper

        #self.hsv_lower_lower = (84,30,55)
        #self.hsv_lower_upper = (140,255,255)
        #self.hsv_upper_lower = self.hsv_lower_lower
        #self.hsv_upper_upper = self.hsv_lower_upper

        #parameters
        self.focal_length = 993.0               #the focal length of the camera 
        self.real_height = 1.25                 #the real height of the target object
        self.image_center = (635.08, 469.80)    #the image center of the camera
        self.rectified = False                  #is the camera using the rectified image
        self.show_color_mask = False            #publish the color mask

        self.pub = rospy.Publisher('dist_color_mask', Image, queue_size=10)
        self.heading_pub = rospy.Publisher('heading', Float64, queue_size=10)
        self.avg_pub = rospy.Publisher('average_heading', Float64, queue_size=10)
        self.subsample_ratio = 0.25

        #for heading averaging
        self.current_sum = 0
        self.count = 0
        self.current_slope = 0
        self.last_point = 0
        self.current_point = 0
        self.average = 0

    def find_position(self, focal_length, real_height, image):
        """ 
        Finds the distance from the camera to a target of known height
        as well as the target's offset from the camera in the x and y planes

        params:
            focal_length (float): the focal length of the camera
            real_height (float): the real world height of the target object 
                 standing in an upright position
            image: the cv2 image from which to seach for the target object
        returns:
            dist, (x_offset, y_offset) 
            the estimated distance between the target object and
            and the camera and the target's offset from image center
        """
        height_px, offset_px = self.process_image(image)
        if height_px == 0:
            return 0, (0,0)
        dist = (focal_length * real_height) / height_px
        x_off = (offset_px[0] * dist)/focal_length
        y_off = (offset_px[1] * dist)/focal_length
        return (dist, (x_off, y_off))

    def process_image(self, image):
        """
        Searches for the target object and returns its height and offset in pixels or
        (0, (0,0)) if not found

        returns (height, (x_offset, y_offset))
        """
        #Resize and blur the image, put into HSV color scale, and create an image mask 
        img_small = cv2.resize(image, None, fx=self.subsample_ratio, fy=self.subsample_ratio, interpolation=cv2.INTER_LINEAR)        
        img_blur = cv2.GaussianBlur(img_small, (5,5), 0)
        img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        mask_l = cv2.inRange(img_hsv, self.hsv_lower_lower, self.hsv_lower_upper)
        mask_u = cv2.inRange(img_hsv, self.hsv_upper_lower, self.hsv_upper_upper)
        mask = cv2.bitwise_or(mask_l, mask_u)
        
        #Publish the mask          
        mask_bgr8 = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        bridge = CvBridge()
        cv_mask = bridge.cv2_to_imgmsg(mask_bgr8, encoding='bgr8')
        self.pub.publish(cv_mask)

        #find the largest contour of the mask or return 0 if target is not there
        img, cnts, cnt_hier = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            return (0, (0,0))        
        c = max(cnts, key=cv2.contourArea)

        #find the height of the target object and its center using minAreaRect
        rect = cv2.minAreaRect(c)        
        height_px =  rect[1][1] / self.subsample_ratio
        target_x =   rect[0][0] / self.subsample_ratio
        target_y =   rect[0][1] / self.subsample_ratio
        offset_px = (target_x - self.image_center[0]) , -1.0*(target_y - self.image_center[1])
        
        #NOTE!! When using a ball shaped object, use minEnclosingCircle and the circle diameter
        #enc_circle = 2 * cv2.minEnclosingCircle(c)[1]
        #height_px = 2 * enc_circle[1]
        #offset_px = (enc_circle[0][0] - self.image_center[0]) , -1*(enc_circle[0][1] - self.image_center[1])

        return height_px, offset_px

    def set_rectified(self, rectified):
        self.rectified = rectified

    def show(self, show):
        self.show_color_mask = show

    def run(self, ros_data):
        """Run the Distance Tracker with the input from the camera"""
        bridge = CvBridge()
        if self.rectified:
            cv_image = bridge.imgmsg_to_cv2(ros_data, desired_encoding='passthrough')
        else:
            cv_image = bridge.compressed_imgmsg_to_cv2(ros_data, desired_encoding='passthrough')
        
        dist, offset = self.find_position(self.focal_length, self.real_height, cv_image)
        
        print("EST DISTANCE: " + str(dist) + ' inches')
        print("EST OFFSET: " + str(offset) + ' inches')
        print("--------------")

        #heading averaging
        self.last_point = self.current_point
        self.current_point = offset[0]
        new_slope = (self.current_point - self.last_point)
        if new_slope != 0:
            new_slope = new_slope / abs(new_slope)
        if self.current_slope == 1:
            if new_slope <= 0:
                self.average = self.current_sum / self.count
                self.current_sum = 0
                self.count = 0
        self.current_sum += self.current_point
        self.count += 1
        self.current_slope = new_slope

        #publish the heading and the average heading
        self.heading_pub.publish(offset[0])
        self.avg_pub.publish(self.average)

if __name__ == '__main__':
    rospy.init_node('dist_tracker', anonymous=True)
    tracker = DistanceTracker()    
    if (len(sys.argv) > 1):
        if ("--rectified" in sys.argv):
            tracker.set_rectified(True)
            rospy.Subscriber('/raspicam_node/image_rect_color', Image, tracker.run)
        if ("--show" in sys.argv):
            tracker.show(True)
    else:
        rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, tracker.run)
    rospy.spin()
