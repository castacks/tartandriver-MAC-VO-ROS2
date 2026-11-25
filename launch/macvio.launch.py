from typing import Any

from launch import Action, LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace

# Adapter defaults per platform; an empty launch argument selects these
PLATFORMS = {
    "yamaha": {
        "namespace": "",
        "reference_topic": "/localization/super_odometry",
        "output_topic": "/superodometry/integrated_to_init",
        "world_frame": "sensor_init",
        "base_frame": "vehicle",
    },
    "stallion": {
        "namespace": "crl_rzr",
        "reference_topic": "liopard/fused/odometry",
        "output_topic": "macvio/odometry",
        "world_frame": "crl_rzr/odom",
        "base_frame": "crl_rzr/base_link",
    },
}
ADAPTER_ARGS = ("reference_topic", "output_topic", "world_frame", "base_frame")
LAUNCH_ARGS = ("platform", "modality", "config_file", "pose_source", "use_sim_time", *ADAPTER_ARGS)


def resolve(arguments: dict[str, str]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Launch arguments -> (namespace, MACVIO node params, adapter params)."""
    platform = PLATFORMS[arguments["platform"]]
    # config_file is the deploy's older argument; it now names a profile yaml
    profile = arguments["config_file"] or f"{arguments['platform']}_{arguments['modality']}"
    use_sim_time = arguments["use_sim_time"].lower() == "true"
    adapter = {name: arguments[name] or platform[name] for name in ADAPTER_ARGS}
    adapter.update(pose_source=arguments["pose_source"], use_sim_time=use_sim_time)
    return platform["namespace"], {"profile": profile, "use_sim_time": use_sim_time}, adapter


def _launch_nodes(context: LaunchContext) -> list[GroupAction]:
    arguments = {name: LaunchConfiguration(name).perform(context) for name in LAUNCH_ARGS}
    namespace, macvio, adapter = resolve(arguments)
    actions: list[Action] = [PushRosNamespace(namespace)] if namespace else []
    actions += [
        Node(package="mac_ros2", executable="MACVIO", name="macvio_node", output="screen", parameters=[macvio]),
        Node(
            package="mac_ros2",
            executable="localization_adapter",
            name="localization_adapter",
            output="screen",
            parameters=[adapter],
        ),
    ]
    return [GroupAction(actions)]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("platform", default_value="yamaha", choices=list(PLATFORMS)),
            DeclareLaunchArgument("modality", default_value="thermal", choices=["thermal", "rgb"]),
            DeclareLaunchArgument("config_file", default_value="", description="Profile yaml overriding platform/modality"),
            DeclareLaunchArgument("pose_source", default_value="macvio", choices=["super_odometry", "macvio"]),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            *(
                DeclareLaunchArgument(name, default_value="", description="Adapter override; empty uses the platform default")
                for name in ADAPTER_ARGS
            ),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
