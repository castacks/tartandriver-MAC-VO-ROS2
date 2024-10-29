colcon build
source install/local_setup.bash

ros2 run MACVO_ROS2 MACVO --config /home/airlab/workspace/MAC-SLAM-ROS2/MACVO_ROS2/config/ZedConfig.yaml
ros2 run macvo_vis macvo_vis

ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedx config_path:=/home/airlab/workspace/MAC-SLAM-ROS2/zed_config.yaml
