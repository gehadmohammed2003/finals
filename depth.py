#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from ultralytics import YOLO
import message_filters  # Import message_filters for synchronization

# Load the YOLO model (Make sure the path is correct)
model = YOLO("/home/gehadatia/catkin_ws/src/kinect_pkg/model/best3.pt")

# Initialize the CvBridge
bridge = CvBridge()

# Function to detect the ball using YOLO and retrieve its 3D position
def detect_ball(cv_image, depth_image, pub_3d):
    # Run the YOLO prediction
    results = model.predict(source=cv_image, show=True)

    # Iterate through detection results
    for result in results:
        if result.boxes is not None:  # Check if there are any detected boxes
            for box in result.boxes:
                class_id = int(box.cls[0])  # Get the class ID
                class_name = result.names[class_id]  # Get the class name from the class ID

                if class_name == 'ball':  # Assuming 'ball' is the label in your model
                    # Get the bounding box coordinates
                    x, y, w, h = box.xywh[0]
                    x = int(x)
                    y = int(y)

                    # Get the depth of the detected ball's center point
                    depth = depth_image[y, x]  # Get the depth at the ball's center

                    if depth != 0:  # Ensure the depth value is valid
                        # Convert 2D (x, y) and depth to 3D coordinates
                        ball_coords_3D = calculate_3D_position(x, y, depth)
                        rospy.loginfo(f"Ball detected at 3D coordinates: {ball_coords_3D}")
                        publish_ball_location(ball_coords_3D, pub_3d)
                    else:
                        rospy.loginfo("No valid depth value for the detected ball.")

# Function to calculate the 3D position from 2D coordinates and depth
def calculate_3D_position(x, y, depth):
    # Kinect V1 intrinsic parameters (adjust if necessary)
    fx = 525.0  # Focal length in x axis (in pixels)
    fy = 525.0  # Focal length in y axis (in pixels)
    cx = 319.5  # Principal point x-coordinate (center of the image)
    cy = 239.5  # Principal point y-coordinate (center of the image)

    # Convert 2D pixel coordinates (x, y) and depth to 3D coordinates (X, Y, Z)
    Z = depth  # Depth in meters
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    return X, Y, Z

# Function to publish the ball's 3D location
def publish_ball_location(coords_3D, pub_3d):
    point_msg = Point()
    point_msg.x = coords_3D[0]
    point_msg.y = coords_3D[1]
    point_msg.z = coords_3D[2]
    pub_3d.publish(point_msg)

# Callback to process synchronized RGB and Depth images
def kinect_callback(rgb_msg, depth_msg, pub_3d):
    # Convert ROS Image messages to OpenCV format
    rgb_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
    depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
    
    # Display the RGB image
    cv2.imshow("Kinect RGB Image", rgb_image)
    cv2.waitKey(1)

    # Detect the ball and get the 3D position
    detect_ball(rgb_image, depth_image, pub_3d)

# Function to subscribe to both RGB and Depth images from the Kinect
def kinect_subscriber():
    rospy.init_node('kinect_subscriber', anonymous=True)

    # Set up the publisher for 3D ball location (on /ball_position topic)
    pub_3d = rospy.Publisher('/ball_position', Point, queue_size=10)

    # Subscribe to both RGB and Depth image topics
    rgb_sub = message_filters.Subscriber('/camera/rgb/image_raw', Image)
    depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)

    # Synchronize the two topics
    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
    ts.registerCallback(lambda rgb_msg, depth_msg: kinect_callback(rgb_msg, depth_msg, pub_3d))

    rospy.spin()

if __name__ == '__main__':
    try:
        kinect_subscriber()
    except rospy.ROSInterruptException:
        pass
