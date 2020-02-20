import rospy
import cv2
import numpy as np
import socket
import sys
import argparse

from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2

from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from rgb_filter.cfg import RGBFilterConfig

parser = argparse.ArgumentParser()
parser.add_argument('--num_photos', default=1)
parser.add_argument('--name', default='zdj')


class make_photo:
    def __init__(self, number_of_photos, name):
        self.bridge = CvBridge()
        self.number_of_photos = int(number_of_photos)
        self.name = name
        self.subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.dptCallback)
        self.counter = 0

    def dptCallback(self, data):
        img = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        cv2.imwrite(self.name + str(self.counter + 1) + '.png', img)
        print self.name + str(self.counter + 1) + ' created'
        cv2.waitKey(1)
        self.counter = self.counter + 1
        if self.counter >= self.number_of_photos:
            self.subscriber.unregister()
            rospy.signal_shutdown('Over')


rospy.init_node('Cameras')
args = parser.parse_args()
MakePhoto = make_photo(args.num_photos, args.name)
rospy.spin()
