import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge



class MonoToBGR(Node):
    def __init__(self):
        super().__init__('mono_to_bgr')

        self.bridge = CvBridge()

        # Right
        self.sub_right = self.create_subscription(
            Image,
            '/multisense/right/image_rect',
            self.callback_right,
            10
        )
        self.pub_right = self.create_publisher(
            Image,
            '/macvo/right/image_rect_bgr8',
            10
        )

        # Left
        self.sub_left = self.create_subscription(
            Image,
            '/multisense/left/image_rect',
            self.callback_left,
            10
        )
        self.pub_left = self.create_publisher(
            Image,
            '/macvo/left/image_rect_bgr8',
            10
        )

    def convert_and_publish(self, msg, publisher, label):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            bgr_msg = self.bridge.cv2_to_imgmsg(bgr_image, encoding='bgr8')
            bgr_msg.header = msg.header  # Preserve timestamp and frame_id
            publisher.publish(bgr_msg)
        except Exception as e:
            self.get_logger().error(f'Error converting {label}: {e}')

    def callback_right(self, msg):
        self.convert_and_publish(msg, self.pub_right, 'right')

    def callback_left(self, msg):
        self.convert_and_publish(msg, self.pub_left, 'left')


def main():
    rclpy.init()
    node = MonoToBGR()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
