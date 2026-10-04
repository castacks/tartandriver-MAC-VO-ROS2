from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "modality",
                default_value="thermal",
                choices=["thermal", "rgb"],
                description="Camera profile selected when config_file is empty",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value="",
                description="Optional ROS bridge config overriding the modality profile",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use the ROS simulation clock",
            ),
            DeclareLaunchArgument(
                "pose_source",
                default_value="super_odometry",
                choices=["super_odometry", "macvio"],
                description="Localization source exposed to the autonomy stack",
            ),
            Node(
                package="mac_ros2",
                executable="MACVIO",
                name="macvio_node",
                output="screen",
                parameters=[
                    {"modality": LaunchConfiguration("modality")},
                    {"config_file": LaunchConfiguration("config_file")},
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
            ),
            Node(
                package="mac_ros2",
                executable="localization_adapter",
                name="localization_adapter",
                output="screen",
                parameters=[
                    {"pose_source": LaunchConfiguration("pose_source")},
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
            ),
        ]
    )
