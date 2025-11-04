import torch
from pathlib import Path
from typing import TYPE_CHECKING
import os, sys
import torch
import pypose as pp
import rerun as rr

from message_filters import ApproximateTimeSynchronizer

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mac_slam'))
sys.path.insert(0, src_path)

from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

if TYPE_CHECKING:
    # To make static type checker happy : )
    from mac_slam.Src.Odometry.MACVO import MACVO
    from mac_slam.Src.DataLoader import StereoFrameData, StereoData, ScaleFrame, SmartResizeFrame
    from mac_slam.Src.Utility.Config import load_config, asNamespace
    from mac_slam.Src.Utility.Visualize import fig_plt, rr_plt
    from mac_slam.Src.Utility.PrettyPrint import print_as_table, ColoredTqdm, Logger
else:
    from Src.Odometry.MACVO import MACVO                
    from Src.DataLoader import StereoFrameData, StereoData, ScaleFrame, SmartResizeFrame
    from Src.Utility.Config import load_config, asNamespace
    from Src.Utility.Visualize import fig_plt, rr_plt
    from Src.Utility.PrettyPrint import print_as_table, ColoredTqdm, Logger


def VisualizeRerunCallback(frame: StereoFrameData, system: MACVO):#, pb: ColoredTqdm):
    if system.prev_frame is None: return

    frame = system.prev_frame[0]
    rr_plt.set_time_sequence(frame.frame_idx)

    if frame.frame_idx > 0:
        rr_plt.log_trajectory("/world/est", pp.SE3(system.graph.frames.data["pose"].data))

    rr_plt.log_camera("/world/macvo/cam", pp.SE3(system.graph.frames.data["pose"][-1]), system.graph.frames.data["K"][-1])
    rr_plt.log_image ("/world/macvo/cam", frame.stereo.imageL[0].permute(1, 2, 0))

    right_pose = pp.identity_SE3().tensor()
    right_pose[1] += system.graph.frames.data["baseline"][-1]
    rr_plt.log_camera("/world/macvo/right/cam", right_pose, system.graph.frames.data["K"][-1])
    rr_plt.log_image ("/world/macvo/right/cam", frame.stereo.imageR[0].permute(1, 2, 0))

    depth = system.prev_frame[-1].depth
    rr_plt.log_depth("/world/macvo/cam/depth", depth)

    map_points = system.graph.get_frame2map(system.graph.frames[-2:-1])
    rr_plt.log_points("/world/point_cloud", map_points.data["pos_Tw"], map_points.data["color"], map_points.data["cov_Tw"], "sphere")

    vo_keypoints = system.graph.get_frame2match(system.graph.frames[-1:])
    vo_points    = system.graph.get_match2point(vo_keypoints)
    rr_plt.log_points("/world/vo_tracking", vo_points.data["pos_Tw"], vo_points.data["color"], vo_points.data["cov_Tw"], "sphere")
    rr_plt.log_keypoints("world/macvo/cam", vo_keypoints.data["pixel2_uv"])

class MACVONode():

    def __init__(self, config_fp, device) -> None:
        
        if isinstance(config_fp, dict):
            cfg = asNamespace(config_fp)
            config_dict = config_fp
        else:
            cfg, config_dict = load_config(path=Path(config_fp))
        self.frame_id = 0

        self.camera = cfg.Camera
        self.device = device
        self.useRR = getattr(cfg, 'useRR', False)

        if self.useRR:
            rr_plt.default_mode = "rerun"
            rr_plt.init_connect("offroad macvo")

        # Set up MACVO odometry
        original_cwd = os.getcwd()
        try:
            os.chdir(Path(__file__).resolve().parent)
            self.odometry = MACVO[StereoFrameData].from_config(cfg)
        finally:
            os.chdir(original_cwd)

        self.frame_fn = SmartResizeFrame({"height": 272, "width": 512, "interp": "nearest"})
        # self.frame_fn = ScaleFrame(dict(scale_u=2, scale_v=2, interp='nearest')) #TODO: change to config arg, not actually 2.

        self.time = None
        self.prev_time = None
        self.coord_frame = "macvo_initial_pose"


    def publish_data(self, system: MACVO):
        # Latest pose       
        pose    = torch.tensor(system.graph.frames.data["pose"][-1], device=self.device)
        time_ns = int(system.graph.frames.data["time_ns"][-1].item())
        
        # Latest map
        if system.mapping:
            points = system.graph.get_frame2map(system.graph.frames[-2:-1])
        else:
            points = system.graph.get_match2point(system.graph.get_frame2match(system.graph.frames[-1:]))


        points = {
            'pos_Tc': torch.tensor(points.data['pos_Tc'], device=self.device),
            'cov_Tc': torch.tensor(points.data['cov_Tw'], device=self.device), #mac called cov_Tc as cov_Tw
            'color': torch.tensor(points.data['color'], device=self.device)
        }
        feature_keys = self.output_feature_keys
        return pose, points, time_ns, feature_keys


    def receive_stereo(self, imageL, imageR, imageLColor, timestamp) -> None:
        
        time_ns = int(timestamp * 1e9)
        self.img_timestamp = time_ns

        # Create a frame
        stereo_frame = self.frame_fn(StereoFrameData(
            idx    =torch.tensor([self.frame_id], dtype=torch.long),
            time_ns=[time_ns],
            stereo =StereoData(
                T_BS=pp.identity_SE3(1, dtype=torch.float64), 
                
                K   =torch.tensor([[
                    [self.camera.fx, 0.            , self.camera.cx],
                    [0.            , self.camera.fy, self.camera.cy],
                    [0.            , 0.            , 1.            ]
                ]]), 
                baseline=torch.tensor([self.camera.bl]),
                time_ns=[time_ns],
                height=imageL.image.shape[0],
                width=imageL.image.shape[1],
                imageL= imageL.image.permute(2, 0, 1).unsqueeze(0),
                imageR= imageR.image.permute(2, 0, 1).unsqueeze(0),
                # imageLColor = imageLColor.image.permute(2, 0, 1).unsqueeze(0),
            )   
        ))

        self.odometry.run(stereo_frame)
        self.frame_id += 1

        if self.useRR:
            VisualizeRerunCallback(stereo_frame, self.odometry)
            rr_plt.log_trajectory("/world/est", pp.SE3(self.odometry.graph.frames.data["pose"].data))
            try:
                rr_plt.log_points    ("/world/point_cloud", 
                                        self.odometry.get_map().map_points.data["pos_Tw"].data,
                                        self.odometry.get_map().map_points.data["color"].data,
                                        self.odometry.get_map().map_points.data["cov_Tw"].data,
                                        "color")
            except RuntimeError:
                Logger.write("warn", "Unable to log full pointcloud - is mapping mode on?")
    
    @property
    def output_feature_keys(self):
        labels = ["r", "g", "b"] + [f"cov_{i}" for i in range(1, 10)]
        metainfo = ["raw"] * 3 + ["macvo"] * 9

        return FeatureKeyList(
            label=labels,
            metainfo=metainfo
        )