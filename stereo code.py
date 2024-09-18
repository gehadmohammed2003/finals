#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO

class TortaBallDetectorNode:
    def __init__(self):
        rospy.init_node('torta_ball_detector_node')
        
        # Load your trained YOLO model
        self.model = YOLO('/path/to/your/trained/yolo/model.pt')
        
        self.bridge = CvBridge()
        
        # Subscribers
        self.left_image_sub = rospy.Subscriber("/stereo/left/image_raw", Image, self.left_image_callback)
        self.right_image_sub = rospy.Subscriber("/stereo/right/image_raw", Image, self.right_image_callback)
        self.left_camera_info_sub = rospy.Subscriber("/stereo/left/camera_info", CameraInfo, self.left_camera_info_callback)
        self.right_camera_info_sub = rospy.Subscriber("/stereo/right/camera_info", CameraInfo, self.right_camera_info_callback)
        
        # Publisher
        self.ball_pub = rospy.Publisher("/ball_position", list, queue_size=10)
        
        self.left_image = None
        self.right_image = None
        self.left_camera_info = None
        self.right_camera_info = None
        
        self.stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=5)
        
        # Red line parameters (adjust these based on your field setup)
        self.red_line_y = 1.5  # Y-coordinate of the red line in meters

    def left_image_callback(self, msg):
        self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process_images()

    def right_image_callback(self, msg):
        self.right_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def left_camera_info_callback(self, msg):
        self.left_camera_info = msg

    def right_camera_info_callback(self, msg):
        self.right_camera_info = msg

    def process_images(self):
        if self.left_image is None or self.right_image is None or \
           self.left_camera_info is None or self.right_camera_info is None:
            return

        # Convert images to grayscale for stereo processing
        left_gray = cv2.cvtColor(self.left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(self.right_image, cv2.COLOR_BGR2GRAY)

        # Compute disparity
        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Detect balls using YOLO on the left image
        results = self.model(self.left_image)
        
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
                    
                    # Get disparity at the center of the detected ball
                    d = disparity[center_y, center_x]
                    
                    if d > 0:
                        # Convert disparity to depth
                        depth = (self.left_camera_info.P[0] * self.right_camera_info.P[3]) / (d * self.left_camera_info.P[0])
                        
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
        x = (pixel_x - self.left_camera_info.K[2]) * depth / self.left_camera_info.K[0]
        y = (pixel_y - self.left_camera_info.K[5]) * depth / self.left_camera_info.K[4]
        return x, y, depth

if __name__ == '__main__':
    try:
        node = TortaBallDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass