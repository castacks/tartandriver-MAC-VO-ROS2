import torch
from pathlib import Path
from typing import TYPE_CHECKING
import os, sys
import torch
import pypose as pp
import rerun as rr

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, src_path)

from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

if TYPE_CHECKING:
    # To make static type checker happy : )
    from src.Odometry.MACVO import MACVO
    from src.DataLoader import StereoFrame, StereoData, ScaleFrame, SmartResizeFrame
    from src.Utility.Config import load_config
else: #This one occurs!
    from torch.utils.data import DataLoader #changed
    from Odometry.MACVO import MACVO                
    from DataLoader import StereoFrame, StereoData, ScaleFrame, SmartResizeFrame
    from Utility.Config import load_config
    from Utility.Visualize import fig_plt, rr_plt
    from Utility.PrettyPrint import print_as_table, ColoredTqdm, Logger


def VisualizeRerunCallback(frame: StereoFrame, system: MACVO):#, pb: ColoredTqdm):
    rr.set_time_sequence("frame_idx", frame.frame_idx)
    
    # Non-key frame does not need visualization
    if system.graph.frames.data["need_interp"][-1]: return
    
    if frame.frame_idx > 0:    
        rr_plt.log_trajectory("/world/est", pp.SE3(system.graph.frames.data["pose"].tensor))
    
    rr_plt.log_camera("/world/macvo/cam_left", pp.SE3(system.graph.frames.data["pose"][-1]), system.graph.frames.data["K"][-1])
    rr_plt.log_image ("/world/macvo/cam_left", frame.stereo.imageL[0].permute(1, 2, 0))
    
    map_points = system.graph.get_frame2map(system.graph.frames[-1:])
    rr_plt.log_points("/world/point_cloud", map_points.data["pos_Tw"], map_points.data["color"], map_points.data["cov_Tw"], "sphere")
    
    vo_points  = system.graph.get_match2point(system.graph.get_frame2match(system.graph.frames[-1:]))
    rr_plt.log_points("/world/vo_tracking", vo_points.data["pos_Tw"], vo_points.data["color"], vo_points.data["cov_Tw"], "sphere")
    

class MACVONode():

    def __init__(self, config_fp, device) -> None:
        
        cfg, _ = load_config( path=Path(config_fp) )
        self.frame_id = 0
        self.camera = cfg.Camera
        self.device = device

        if False:
            rr_plt.default_mode = "rerun"
            rr_plt.init_connect(self.project_name)
            fig_plt.default_mode = "none" #"image" if args.saveplt else "none"

        # Set up MACVO odometry
        original_cwd = os.getcwd()
        try:
            os.chdir(Path(__file__).resolve().parent)
            self.odometry = MACVO[StereoFrame].from_config(cfg)
        finally:
            os.chdir(original_cwd)

        # self.frame_fn = SmartResizeFrame({"height": 320, "width": 320, "interp": "bilinear"}) Yifei: SmartResizeFrame({"height": 272, "width":512, "interp":"nearest"})
        self.frame_fn = ScaleFrame(dict(scale_u=2, scale_v=2, interp='nearest')) #TODO: change to config arg, not actually 2.

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
            'cov_Tc': torch.tensor(points.data['cov_Tc'], device=self.device),
            'color': torch.tensor(points.data['color'], device=self.device)
        }
        feature_keys = self.output_feature_keys
        return pose, points, time_ns, feature_keys


    def receive_stereo(self, imageL, imageR, imageLColor, timestamp) -> None:
        
        time_ns = int(timestamp * 1e9)
        self.img_timestamp = time_ns

        # Create a frame
        stereo_frame = self.frame_fn(StereoFrame(
            idx    =[self.frame_id],
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
                imageLColor = imageLColor.image.permute(2, 0, 1).unsqueeze(0),
            )   
        ))

        self.odometry.run(stereo_frame)
        self.frame_id += 1

        if False:
            VisualizeRerunCallback(stereo_frame, self.odometry)
            if self.frame_id == 3:
                rr_plt.log_trajectory("/world/est", pp.SE3(self.odometry.graph.frames.data["pose"].tensor))
                try:
                    rr_plt.log_points    ("/world/point_cloud", 
                                            self.odometry.get_map().map_points.data["pos_Tw"].tensor,
                                            self.odometry.get_map().map_points.data["color"].tensor,
                                            self.odometry.get_map().map_points.data["cov_Tw"].tensor,
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