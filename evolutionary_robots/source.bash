#!/usr/bin/env bash

# Navigate to the correct directory
cd ./../catkin_ws

# The catkin workspace directory should have been built
source devel/setup.bash

# Set the environment variable to use our Gazebo Models
export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}:$PWD/src/evolutionary_robotics/models/

# Navigate back to where we came from
cd ./../evolutionary_robots

# The installation would be complete now
# The user can run the exercise

