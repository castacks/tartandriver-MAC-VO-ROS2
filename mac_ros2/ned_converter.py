# ...existing code...
import numpy as np
import rclpy
import math
from rclpy.node import Node
from sensor_msgs.msg import PointCloud, ChannelFloat32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point32


class NEDConverterNode(Node):
    def __init__(self):
        super().__init__('ned_converter_node')
        self.pc_sub = self.create_subscription(PointCloud, '/macvo/pointcloud', self.pc_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/macvo/pose', self.odom_callback, 10)
        self.pc_pub = self.create_publisher(PointCloud, '/macvo/flu_pointcloud', 10)
        self.odom_pub = self.create_publisher(Odometry, '/macvo/flu_odometry', 10)

    def pc_callback(self, msg):
        new_msg = PointCloud()
        new_msg.header = msg.header
        new_msg.points = []
        for pt in msg.points:
            # Macvo's NED: x=forward, y=right, z=down to offroad stack's FLU: x=forward, y=left, z=up 
            new_pt = Point32()
            pt.x = pt.x
            pt.y = -pt.y
            pt.z = -pt.z

            # Rotate 12deg clockwise about Y
            degrees = 12.9
            radians = math.radians(degrees)
            cos_theta = math.cos(radians)
            sin_theta = math.sin(radians)

            # Rotation around y axis:
            # x' =  cosθ * x + sinθ * z
            # y' =  y
            # z' = -sinθ * x + cosθ * z
            new_pt.x =  cos_theta * pt.x + sin_theta * pt.z
            new_pt.y =  pt.y
            new_pt.z = -sin_theta * pt.x + cos_theta * pt.z

            new_msg.points.append(new_pt)

        # Copy channels directly (order is preserved)
        new_msg.channels = []
        for ch in msg.channels:
            new_ch = ChannelFloat32()
            new_ch.name = ch.name
            new_ch.values = list(ch.values)  # Copy all values as-is
            new_msg.channels.append(new_ch)

        self.pc_pub.publish(new_msg)

    def odom_callback(self, msg):
        # Convert Odometry pose and twist from NED to FLU
        new_msg = Odometry()
        new_msg.header = msg.header
        new_msg.child_frame_id = msg.child_frame_id

        new_msg.pose.pose.position.x =   msg.pose.pose.position.x
        new_msg.pose.pose.position.y = - msg.pose.pose.position.y
        new_msg.pose.pose.position.z = - msg.pose.pose.position.z

        q = msg.pose.pose.orientation
        new_msg.pose.pose.orientation.x =   q.x
        new_msg.pose.pose.orientation.y = - q.y
        new_msg.pose.pose.orientation.z = - q.z
        new_msg.pose.pose.orientation.w =   q.w

        new_msg.twist.twist.linear.x =   msg.twist.twist.linear.x
        new_msg.twist.twist.linear.y = - msg.twist.twist.linear.y
        new_msg.twist.twist.linear.z = - msg.twist.twist.linear.z

        new_msg.twist.twist.angular.x =   msg.twist.twist.angular.x
        new_msg.twist.twist.angular.y = - msg.twist.twist.angular.y
        new_msg.twist.twist.angular.z = - msg.twist.twist.angular.z

        # Rotate 12deg clockwise about Y for position and velocity
        degrees = 12.9
        radians = math.radians(degrees)
        cos_theta = math.cos(radians)
        sin_theta = math.sin(radians)

        # Rotate position
        x = new_msg.pose.pose.position.x
        y = new_msg.pose.pose.position.y
        z = new_msg.pose.pose.position.z
        new_msg.pose.pose.position.x = cos_theta * x + sin_theta * z
        new_msg.pose.pose.position.y = y
        new_msg.pose.pose.position.z = -sin_theta * x + cos_theta * z

        # Rotate linear velocity
        lx = new_msg.twist.twist.linear.x
        ly = new_msg.twist.twist.linear.y
        lz = new_msg.twist.twist.linear.z
        new_msg.twist.twist.linear.x = cos_theta * lx + sin_theta * lz
        new_msg.twist.twist.linear.y = ly
        new_msg.twist.twist.linear.z = -sin_theta * lx + cos_theta * lz

        # Rotate angular velocity
        ax = new_msg.twist.twist.angular.x
        ay = new_msg.twist.twist.angular.y
        az = new_msg.twist.twist.angular.z
        new_msg.twist.twist.angular.x = cos_theta * ax + sin_theta * az
        new_msg.twist.twist.angular.y = ay
        new_msg.twist.twist.angular.z = -sin_theta * ax + cos_theta * az

def main(args=None):
    rclpy.init()
    node = NEDConverterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

