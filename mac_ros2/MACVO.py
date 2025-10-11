# deprecated in offroad
import rclpy
import torch
import pypose as pp

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

from message_filters import ApproximateTimeSynchronizer, Subscriber
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time

from pathlib import Path
from typing import TYPE_CHECKING
import os, sys
import argparse
import logging

from .MessageFactory import to_stamped_pose, from_image, to_pointcloud, to_image
from rclpy.clock import Clock, ClockType
from rclpy.parameter import Parameter

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mac_slam'))
sys.path.insert(0, src_path)

if TYPE_CHECKING:
    # To make static type checker happy : )
    from mac_slam.Odometry.MACVO import MACVO
    from mac_slam.DataLoader import StereoFrame, StereoData, SmartResizeFrame
    from mac_slam.Utility.Config import load_config
    from mac_slam.Utility.PrettyPrint import Logger
    from mac_slam.Utility.Timer import Timer
else:
    from torch.utils.data import DataLoader #changed
    from Odometry.MACVO import MACVO                
    from DataLoader import StereoFrame, StereoData, SmartResizeFrame
    from Utility.Config import load_config
    from Utility.PrettyPrint import Logger
    from Utility.Timer import Timer

PACKAGE_NAME = "mac_ros2"

# MAC-VO Node

# def setup_macvo(config_fp):
            
#             # Start MAC-VO
#         import pdb; pdb.set_trace()
#         from pathlib import Path

        
#         node = MACVONode( 
#             config=Path(args.config), #use config, similar to core/mission_manager
#             imageL_subscribe="/macvo/left/image_rect_bgr8", #"/multisense/left/image_rect_color",
#             imageR_subscribe="/macvo/right/image_rect_bgr8",
#             pose_publish    ="/macvo/pose",
#             map_pc_publish  ="/macvo/pointcloud",
#             imageL_publish  ="/macvo/img",
#             )
#         cfg, _ = load_config(Path(config_fp))
#         node.frame_id, node.camera = 0, cfg.Camera
        
#         # Load model from the resources directory automatically.
#         original_cwd = os.getcwd()
#         try:
#             os.chdir(get_package_share_directory(PACKAGE_NAME))
#             node.get_logger().info(get_package_share_directory(PACKAGE_NAME))
#             node.odometry = MACVO[StereoFrame].from_config(cfg)
#         finally:
#             os.chdir(original_cwd)
#         # End
#         node.frame_fn = SmartResizeFrame({"height": 320, "width": 320, "interp": "bilinear"})F
#         #

#         node.odometry.register_on_optimize_finish(node.publish_data)
        
#         node.time, node.prev_time = None, None
#         node.coord_frame = "zed_lcam_initial_pose"
        
#         return node #self.odometry #or more like self.



class MACVONode(Node):
    def __init__(
        self, config: Path,
        imageL_subscribe: str,  # Rectified RGB image, expect to receive sensor_msg/image
        imageR_subscribe: str,  # Rectified RGB image, expect to receive sensor_msg/image

        pose_publish: str,      # Estimated pose of Left camera optical center, NED coordinate, geometry_msg/pose_stamped
        map_pc_publish: str,    # Dense mapping pointcloud, sensor_msg/pointcloud 
        imageL_publish: str     # Scaled & cropped RGB image actually received by MAC-VO, sensor_msg/image
    ) -> None:
        super().__init__("macvo")

        self.set_parameters([
            Parameter('use_sim_time', Parameter.Type.BOOL, True)
        ])

        self.get_logger().set_level(logging.INFO)
        self.get_logger().info(f"{os.getcwd()}")

        imageL_recv = Subscriber(self, Image, imageL_subscribe, qos_profile=1)
        imageR_recv = Subscriber(self, Image, imageR_subscribe, qos_profile=1)
        self.stereo_recv = ApproximateTimeSynchronizer([imageL_recv, imageR_recv], queue_size=2, slop=0.1)
        self.stereo_recv.registerCallback(self.receive_stereo)

        self.pose_send = self.create_publisher(Odometry, pose_publish, qos_profile=1) #changed from PoseStamped
        self.map_send  = self.create_publisher(PointCloud, map_pc_publish, qos_profile=1)
        self.img_send  = self.create_publisher(Image, imageL_publish, qos_profile=1)
        
        # Start MAC-VO
        #import pdb; pdb.set_trace()
        cfg, _ = load_config(config)
        self.frame_id, self.camera = 0, cfg.Camera
        
        # Load model from the resources directory automatically.
        original_cwd = os.getcwd()
        try:
            os.chdir(get_package_share_directory(PACKAGE_NAME))
            self.get_logger().info(get_package_share_directory(PACKAGE_NAME))
            self.odometry = MACVO[StereoFrame].from_config(cfg)
        finally:
            os.chdir(original_cwd)
        # End
        self.frame_fn = SmartResizeFrame({"height": 320, "width": 320, "interp": "bilinear"})
        #

        self.odometry.register_on_optimize_finish(self.publish_data)
        
        self.time, self.prev_time = None, None
        self.coord_frame = "zed_lcam_initial_pose"

    def publish_data(self, system: MACVO):
        # Latest pose
        pose    = pp.SE3(system.graph.frames.data["pose"][-1])
        time_ns = int(system.graph.frames.data["time_ns"][-1].item())
        time = Time()
        time.sec = time_ns // 1_000_000_000
        time.nanosec = time_ns % 1_000_000_000
        pose_msg = to_stamped_pose(pose, self.coord_frame, time) 

        
        # Latest map
        if system.mapping:
            points = system.graph.get_frame2map(system.graph.frames[-2:-1])
        else:
            points = system.graph.get_match2point(system.graph.get_frame2match(system.graph.frames[-1:]))

        map_pc_msg = to_pointcloud( #TODO see how frames are defined in MACVO, compared to us
            position  = points.data["pos_Tw"],
            keypoints = None,
            frame_id  = self.coord_frame,
            colors    = points.data["color"],
            time      = time,
        )
        #import pdb; pdb.set_trace()

        #timestamp = # self.get_clock().now().to_msg()
        pose_msg.header.stamp = self.img_timestamp
        map_pc_msg.header.stamp = self.img_timestamp
        
        self.pose_send.publish(pose_msg)
        self.map_send.publish(map_pc_msg)

    def receive_stereo(self, msg_imageL: Image, msg_imageR: Image) -> None:
        self.get_logger().info(f"{self.odometry.graph}")
        imageL, timestamp = from_image(msg_imageL), msg_imageL.header.stamp
        imageR            = from_image(msg_imageR)
        
        self.img_timestamp = timestamp
        #ROS VERSION
        # Instantiate a frame and scale to the desired height & width
        stereo_frame = self.frame_fn(StereoFrame(
            idx    =[self.frame_id],
            time_ns=[timestamp.nanosec],
            stereo =StereoData(
                T_BS=pp.identity_SE3(1),
                K   =torch.tensor([[
                    [self.camera.fx, 0.            , self.camera.cx],
                    [0.            , self.camera.fy, self.camera.cy],
                    [0.            , 0.            , 1.            ]
                ]]), #add a batch dimension
                baseline=torch.tensor([self.camera.bl]),
                time_ns=[timestamp.nanosec],
                height=imageL.shape[0],
                width=imageL.shape[1],
                imageL=torch.tensor(imageL)[..., :3].float().permute(2, 0, 1).unsqueeze(0) / 255.,
                imageR=torch.tensor(imageR)[..., :3].float().permute(2, 0, 1).unsqueeze(0) / 255.,
            )
        ))

        print(stereo_frame)
        #import pdb; pdb.set_trace()

        self.odometry.run(stereo_frame)
        
        # Pose-processing
        self.frame_id += 1
    
    def destroy_node(self):
        self.odometry.terminate()


def main():
    # rclpy.init()
    # args = argparse.ArgumentParser()
    # args.add_argument("--config", type=str, default="./config/ZedConfig.yaml") #TODO check this out
    # args.add_argument("--timing", action="store_true", help="Record timing for system (active Utility.Timer for global time recording)")
    # args = args.parse_args()
    
    rclpy.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/ZedConfig.yaml")
    parser.add_argument("--timing", action="store_true", help="Record timing for system (active Utility.Timer for global time recording)")

    args, unknown = parser.parse_known_args()


    # Optional - setup Timer to see profiling result
    Timer.setup(active=args.timing)
    # Optional End
    
    # Create Node and start running

    #TODO look at mission manager, how it uses config with a launch file
    #1. check if its working running only this node
    #2. update configs...
    node = MACVONode( 
        config=Path(args.config), #use config, similar to core/mission_manager
        imageL_subscribe="/macvo/left/image_rect_bgr8", #"/multisense/left/image_rect_color",
        imageR_subscribe="/macvo/right/image_rect_bgr8",
        pose_publish    ="/macvo/pose",
        map_pc_publish  ="/macvo/pointcloud",
        imageL_publish  ="/macvo/img",
    )
    print('MACVO Node created.')
    rclpy.spin(node)
    # End
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__': main()
