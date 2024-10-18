
## Install
This repo support ros2_humble, you may check the robostack: https://robostack.github.io/GettingStarted.html#__tabbed_1_1

## Submodules
This ros embeding use the `ROS-2` branch of the `MAC-SLAM`

```
git submodule set-branch -b ROS-2 MACVO_ROS2/src
git submodule update --init --remote MACVO_ROS2/src
```

## Pretrained Model

Please follow the https://github.com/MAC-VO/MAC-VO to download the pre-trained model. The default path for the pre-trained model are in `MACVO_ROS2/src/Module`
