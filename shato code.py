#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO

class ShatoBallDetectorNode:
    def __init__(self):
        rospy.init_node('shato_ball_detector_node')
        
        # Load your trained YOLO model
        self.model = YOLO('/path/to/your/trained/yolo/model.pt')
        
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber("/camera/rgb/camera_info", CameraInfo, self.camera_info_callback)
        
        # Publisher
        self.ball_pub = rospy.Publisher("/ball_position", list, queue_size=10)
        
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None
        
        # Red line parameters (adjust these based on your field setup)
        self.red_line_y = 1.5  # Y-coordinate of the red line in meters

    def image_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process_images()

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def process_images(self):
        if self.rgb_image is None or self.depth_image is None or self.camera_info is None:
            return

        # Detect balls using YOLO
        results = self.model(self.rgb_image)
        
        closest_ball = None
        min_distance = float('inf')

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf
                cls = box.cls
                
                if conf > 0.5 and cls == 0:  # Assuming class 0 is blue ball
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Get depth at the center of the detected ball
                    depth = self.depth_image[center_y, center_x]
                    
                    if depth > 0 and not np.isnan(depth):
                        # Convert to 3D coordinates
                        x, y, z = self.pixel_to_3d(center_x, center_y, depth)
                        
                        # Check if the ball is before the red line
                        if y < self.red_line_y:
                            distance = np.sqrt(x**2 + y**2 + z**2)
                            if distance < min_distance:
                                min_distance = distance
                                closest_ball = [x, y]

        if closest_ball:
            self.ball_pub.publish(closest_ball)
            rospy.loginfo(f"Published closest ball position: {closest_ball}")
        else:
            self.ball_pub.publish([0, 0])
            rospy.loginfo("No balls detected or all balls are beyond the red line. Published [0, 0].")

    def pixel_to_3d(self, pixel_x, pixel_y, depth):
        x = (pixel_x - self.camera_info.K[2]) * depth / self.camera_info.K[0]
        y = (pixel_y - self.camera_info.K[5]) * depth / self.camera_info.K[4]
        return x, y, depth

if __name__ == '__main__':
    try:
        node = ShatoBallDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass