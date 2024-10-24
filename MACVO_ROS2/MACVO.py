import rclpy
import torch
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber

from pathlib import Path
from typing import TYPE_CHECKING
from torchvision.transforms.functional import center_crop
import os, sys
import argparse

from .MessageFactory import to_stamped_pose, from_image, to_pointcloud, to_image

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, src_path)
if TYPE_CHECKING:
    # To make static type checker happy : )
    from src.Odometry.MACVO import MACVO
    from src.DataLoader import SourceDataFrame, MetaInfo
    from src.Utility.Config import load_config
    from src.Utility.Visualizer import PLTVisualizer
else:
    from Odometry.MACVO import MACVO                
    from DataLoader import SourceDataFrame, MetaInfo
    from Utility.Config import load_config
    from Utility.Visualizer import PLTVisualizer


class MACVONode(Node):
    def __init__(
        self,
        imageL_topic: str,
        imageR_topic: str,
        pose_topic: str,
        MACVO_config: str,
        point_topic: str | None = None,
        img_stream: str | None = None,
    ):
        super().__init__("macvo")
        self.imageL_sub = Subscriber(self, Image, imageL_topic, qos_profile=1)
        self.imageR_sub = Subscriber(self, Image, imageR_topic, qos_profile=1)
        
        self.sync_stereo = ApproximateTimeSynchronizer(
            [self.imageL_sub, self.imageR_sub], queue_size=2, slop=0.1
        )
        self.sync_stereo.registerCallback(self.receive_frame)
        
        self.pose_pipe  = self.create_publisher(PoseStamped, pose_topic, qos_profile=1)
        
        if point_topic is not None:
            self.point_pipe = self.create_publisher(PointCloud, point_topic, qos_profile=1)
        else:
            self.point_pipe = None
        
        if img_stream is not None:
            self.img_pipes = self.create_publisher(Image, img_stream, qos_profile=1)
        else:
            self.img_pipes = None
        
        cfg, _ = load_config(Path(MACVO_config))
        self.frame_idx  = 0
        self.camera     = cfg.Camera
        self.odometry   = MACVO.from_config(cfg)
        
        self.odometry.register_on_optimize_finish(self.publish_latest_pose)
        self.odometry.register_on_optimize_finish(self.publish_latest_points)
        self.odometry.register_on_optimize_finish(self.publish_latest_stereo)
        
        self.time  , self.prev_time  = None, None
        self.frame = "zed_lcam_initial_pose"
    
    def publish_latest_pose(self, system: MACVO):
        pose = system.gmap.frames.pose[-1]
        frame = self.frame
        time  = self.time if self.prev_time is None else self.prev_time
        assert frame is not None and time is not None
        
        out_msg = to_stamped_pose(pose, frame, time)
        
        self.pose_pipe.publish(out_msg)
   
    def publish_latest_points(self, system: MACVO):
        if self.point_pipe is None: return
        
        latest_frame  = system.gmap.frames[-1]
        latest_points = system.gmap.get_frame_points(latest_frame)
        latest_obs    = system.gmap.get_frame_observes(latest_frame)
        
        frame = self.frame
        time  = self.time if self.prev_time is None else self.prev_time
        assert frame is not None and time is not None
        
        out_msg = to_pointcloud(
            position  = latest_points.position,
            keypoints = latest_obs.pixel_uv,
            frame_id  = frame,
            colors    = latest_points.color,
            time      = time
        )
        self.point_pipe.publish(out_msg)
  
    def publish_latest_stereo(self, system: MACVO):
        if self.img_pipes is None: return
        
        source = system.prev_frame
        if source is None: return
        frame = self.frame
        time  = self.time if self.prev_time is None else self.prev_time
        assert frame is not None and time is not None
        
        msg: Image = to_image(
            (center_crop(source.imageL[0].clone(), [224, 224]).permute(1, 2, 0).numpy() * 255).astype(np.uint8),
            encoding="bgr8", frame_id=frame, time=time
        )
        self.img_pipes.publish(msg)
        
        
    def receive_frame(self, msg_L: Image, msg_R: Image) -> None:
        self.prev_frame, self.prev_time = self.frame, self.time
        
        self.frame        = msg_L.header.frame_id
        imageL, self.time = from_image(msg_L), msg_L.header.stamp
        imageR            = from_image(msg_R)
        self.get_logger().info(f"Receive: Left={imageL.shape}, Right={imageR.shape}, time={self.time}, frame={self.frame}")
        
        # Receive image
        meta=MetaInfo(
            idx=self.frame_idx,
            baseline=self.camera.bl,
            width=self.camera.width,
            height=self.camera.height,
            K=torch.tensor([[self.camera.fx, 0., self.camera.cx],
                            [0., self.camera.fy, self.camera.cy],
                            [0., 0., 1.]])
        )
        frame = SourceDataFrame(
                meta=meta,
                imageL=torch.tensor(imageL)[..., :3].float().permute(2, 0, 1).unsqueeze(0) / 255.,
                imageR=torch.tensor(imageR)[..., :3].float().permute(2, 0, 1).unsqueeze(0) / 255.,
                imu=None,
                gtFlow=None, gtDepth=None, gtPose=None, flowMask=None
            )\
            .scale_image(scale_u=6, scale_v=5)
        
        self.odometry.run(frame)
        self.frame_idx += 1


def main():
    # PLTVisualizer.setup(state=PLTVisualizer.State.SAVE_FILE, save_path=Path("/home/yutian/ros2_ws/.output"))
    rclpy.init()
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="./config/ZedConfig.yaml")
    args = args.parse_args()
    
    node = MACVONode(
        imageL_topic="/zed/zed_node/rgb/image_rect_color",
        imageR_topic="/zed/zed_node/right/image_rect_color",
        pose_topic  ="/macvo/pose",
        point_topic ="/macvo/map",
        img_stream  ="/macvo/img",
        MACVO_config=str(Path(args.config))
    )
    print('MACVO Node created.')
    
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
