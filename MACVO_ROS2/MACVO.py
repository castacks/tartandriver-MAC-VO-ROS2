import rclpy
import numpy as np
import pypose as pp

from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from message_filters import ApproximateTimeSynchronizer, Subscriber

import torch

class MACVONode(Node):
    _name_to_dtypes = {
        "rgb8":    (np.uint8,  3),
        "rgba8":   (np.uint8,  4),
        "rgb16":   (np.uint16, 3),
        "rgba16":  (np.uint16, 4),
        "bgr8":    (np.uint8,  3),
        "bgra8":   (np.uint8,  4),
        "bgr16":   (np.uint16, 3),
        "bgra16":  (np.uint16, 4),
        "mono8":   (np.uint8,  1),
        "mono16":  (np.uint16, 1),

        # for bayer image (based on cv_bridge.cpp)
        "bayer_rggb8":      (np.uint8,  1),
        "bayer_bggr8":      (np.uint8,  1),
        "bayer_gbrg8":      (np.uint8,  1),
        "bayer_grbg8":      (np.uint8,  1),
        "bayer_rggb16":     (np.uint16, 1),
        "bayer_bggr16":     (np.uint16, 1),
        "bayer_gbrg16":     (np.uint16, 1),
        "bayer_grbg16":     (np.uint16, 1),

        # OpenCV CvMat types
        "8UC1":    (np.uint8,   1),
        "8UC2":    (np.uint8,   2),
        "8UC3":    (np.uint8,   3),
        "8UC4":    (np.uint8,   4),
        "8SC1":    (np.int8,    1),
        "8SC2":    (np.int8,    2),
        "8SC3":    (np.int8,    3),
        "8SC4":    (np.int8,    4),
        "16UC1":   (np.uint16,   1),
        "16UC2":   (np.uint16,   2),
        "16UC3":   (np.uint16,   3),
        "16UC4":   (np.uint16,   4),
        "16SC1":   (np.int16,  1),
        "16SC2":   (np.int16,  2),
        "16SC3":   (np.int16,  3),
        "16SC4":   (np.int16,  4),
        "32SC1":   (np.int32,   1),
        "32SC2":   (np.int32,   2),
        "32SC3":   (np.int32,   3),
        "32SC4":   (np.int32,   4),
        "32FC1":   (np.float32, 1),
        "32FC2":   (np.float32, 2),
        "32FC3":   (np.float32, 3),
        "32FC4":   (np.float32, 4),
        "64FC1":   (np.float64, 1),
        "64FC2":   (np.float64, 2),
        "64FC3":   (np.float64, 3),
        "64FC4":   (np.float64, 4)
    }

    def __init__(self, imageL_topic: str, imageR_topic: str, pose_topic: str):
        super().__init__("macvo")
        self.imageL_sub = Subscriber(self, Image, imageL_topic)
        self.imageR_sub = Subscriber(self, Image, imageR_topic)
        
        self.sync_stereo = ApproximateTimeSynchronizer(
            [self.imageL_sub, self.imageR_sub], queue_size=10, slop=0.1
        )
        self.sync_stereo.registerCallback(self.receive_frame)
        
        self.pose_pipe  = self.create_publisher(
            PoseStamped, pose_topic, qos_profile=10
        )
    
    @staticmethod
    def ros2_image_to_numpy(msg: Image) -> np.ndarray:
        if msg.encoding not in MACVONode._name_to_dtypes:
            raise KeyError(f"Unsupported image encoding {msg.encoding}")
        
        dtype_name, channel = MACVONode._name_to_dtypes[msg.encoding]
        dtype = np.dtype(dtype_name)
        dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
        shape = (msg.height, msg.width, channel)
        
        data = np.frombuffer(msg.data, dtype=dtype).reshape(shape)
        data.strides = (msg.step, dtype.itemsize * channel, dtype.itemsize)
        return data
    
    def receive_frame(self, msg_L: Image, msg_R: Image) -> None:
        frame        = msg_L.header.frame_id
        imageL, time = self.ros2_image_to_numpy(msg_L), msg_L.header.stamp
        imageR       = self.ros2_image_to_numpy(msg_R)
        self.get_logger().info(f"Left={imageL.shape}, Right={imageR.shape}")
        
        # Receive image
        pose = pp.identity_SE3()
        
        # Return pose
        out_msg = PoseStamped()
        out_msg.header = Header()
        out_msg.header.stamp = time
        out_msg.header.frame_id = frame
        
        out_msg.pose.position.x = pose[0].item()
        out_msg.pose.position.y = pose[1].item()
        out_msg.pose.position.z = pose[2].item()
        
        out_msg.pose.orientation.x = pose[3].item()
        out_msg.pose.orientation.y = pose[4].item()
        out_msg.pose.orientation.z = pose[5].item()
        out_msg.pose.orientation.w = pose[6].item()
        
        self.pose_pipe.publish(out_msg)


def main():
    rclpy.init()
    
    node = MACVONode(
        imageL_topic="/zed/zed_node/rgb/image_rect_color",
        imageR_topic="/zed/zed_node/right/image_rect_color",
        pose_topic="/macvo/pose"
    )
    print('MACVO Node created.')
    
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
