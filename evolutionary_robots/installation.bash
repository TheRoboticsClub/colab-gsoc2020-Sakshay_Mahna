#!/usr/bin/env bash

# catkin_build the resources
cd ./../catkin_ws
catkin_make

# Navigate back to where we came from
cd ./../evolutionary_robots
