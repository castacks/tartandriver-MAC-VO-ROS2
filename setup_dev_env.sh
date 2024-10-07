source .venv/bin/activate        # Initialize virtual envrionment
colcon build
source install/local_setup.bash

ros2 run MACVO_ROS2 MACVO
