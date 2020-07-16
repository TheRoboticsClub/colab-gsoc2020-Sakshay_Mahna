#!/usr/bin/env bash

# catkin_build the resources
cd ./../catkin_ws
catkin_make

# After the resources are built, they are sourced
source devel/setup.bash

# Set the environment variable to use our Gazebo Models
export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}:$PWD/src/evolutionary_robotics/models/

# Navigate back to where we came from
cd ./../evolutionary_robotics

# The installation would be complete now
# The user can run the exercise

