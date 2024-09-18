#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String  # Correct import for String messages
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("/home/ziad/catkin_ws/src/kinect_pkg/model/best.pt")

# Function to detect a ball and publish a message
def detect_ball(cv_image, pub):
    # Run the YOLO prediction
    results = model.predict(source=cv_image,show=True)
    
    # Iterate through detection results
    for result in results:
        # Assuming 'ball' is the label for the ball in your model
        for det in result:
            if det.label == 'ball':
                # Publish a message when a ball is detected
                pub.publish("Ball detected with confidence: {:.2f}".format(det.conf))
                rospy.loginfo("Ball detected with confidence: {:.2f}".format(det.conf))

# Kinect callback to process images
def kinect_callback(image_msg):
    bridge = CvBridge()
    
    # Convert the ROS Image message to OpenCV format
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
    
    # Display the image
    cv2.imshow("Kinect RGB Image", cv_image)
    cv2.waitKey(1)
    
    # Detect ball and publish detection
    detect_ball(cv_image, kinect_subscriber.publisher)

# Initialize ROS node and set up subscriber and publisher
def kinect_subscriber():
    rospy.init_node('kinect_subscriber', anonymous=True)
    
    # Set up a publisher to publish when a ball is detected
    kinect_subscriber.publisher = rospy.Publisher('ball_detection', String, queue_size=10)
    
    # Subscribe to the Kinect image topic
    rospy.Subscriber('/camera/rgb/image_raw', Image, kinect_callback)
    
    rospy.spin()

if __name__ == '__main__':
    try:
        kinect_subscriber()
    except rospy.ROSInterruptException:
        pass
