"""Super Odometry as the localization adapter's reference: its odometry and TF are renamed so only the adapter publishes them."""
from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import SetRemap
from launch_ros.substitutions import FindPackageShare

# Same renames as the passive playback bag remaps
REMAPS = {
    "/superodometry/integrated_to_init": "/localization/super_odometry",
    "/tf": "/localization/reference_tf",
}


def generate_launch_description() -> LaunchDescription:
    launch_file = PathJoinSubstitution([FindPackageShare("super_odometry"), "launch/learningphysics/yamaha.launch"])
    super_odometry = IncludeLaunchDescription(AnyLaunchDescriptionSource(launch_file))
    return LaunchDescription(
        [GroupAction([*(SetRemap(src=src, dst=dst) for src, dst in REMAPS.items()), super_odometry])]
    )
