---
layout: post
title:  "Week #6: Gazebo Simulation"
date:   2020-07-17 10:30:00 +0530
categories: gazebo-ros
comments: true
---
This week’s task involved working on the Gazebo Simulation of the exercise. 

## Where does this fit in the project?
Robot Simulation provides the best **debugging and development** environment. Atleast, they help us get started with any of our ambitious robot projects. For a student learning robotics, simulations are a great way to learn and build robots. The exercises we are developing will also require the use of simulators. The simulations will allow the student to check the current behaviour of the robot, whether while learning or while testing. By default, ROS uses **Gazebo Simulator** software. The simulation of our exercise is on the same. I also wrote a [blog post on debugging](https://theroboticsclub.github.io/colab-Sakshay_Mahna/2020-01-04-talking-with-robots/) a long time back. Do check it out!

## Outcome of this Week
The Gazebo simulation of the robot is complete. The robot used in the simulator has 8 sensors all around it's body(similar to the Khepera robot used in the D.Floreano experiment). Along with it, the simulation is linked to the Robotics Academy template. Similar to the other exercises, the student can now input his/her code in `MyAlgorithm.py` and the robot will start behaving accordingly.

![Gazebo Simulation](./../assets/gif/simulation.gif)

*Something visual to show from the project*

## Logic of the Code
There were a lot of little subtasks that need to be accomplished before the final simulation was complete. 

- **Development of the Robot Model**: The `jderobot/assets` repository contains various robot models that are used in various projects of JdeRobot be it Robotics Academy, Behaviour Studio and so on! However, there is no robot model that suits to the need of our exercise. Our exercise, needs a robot with 1D range sensors pointing in 8 different directions. So to accomplish this, we selected an existing robot, the iRobot Create, removed the Hokuyo Laser on it and embedded the 8 sonar sensors([ROS Gazebo Plugin](http://osrf-distributions.s3.amazonaws.com/gazebo/api/dev/classgazebo_1_1SensorPlugin.html) for SonarSensor). [This is a good link](https://medium.com/teamarimac/integrating-sonar-and-ir-sensor-plugin-to-robot-model-in-gazebo-with-ros-656fd9452607) to follow for the task.

- **Devlopment of the world file**: The Gazebo world file allows the user to import robot models into a running simulation. After creating the robot model, a map model was also selected to serve as walls, which the robot would avoid collision with. Both of these models, along with a ground plane were put in a world file which is launched using a launch file for the exercise.

- **Interfacing the IR sensors**: As explained above, Robotics Academy currently has no robot that has IR sensors around a robot. Therefore, there is no interface file that collects data from sensors like these and provides it to the user. Hence, an interface was also to be created. The interface, subscribes to the ROS topic that publishes the sensor data and provides it to the user, in the form of an easier interface. The user can then use the sensor data and program the robot accordingly.

- **Connecting the template**: This was pretty simple. For this, we just had to copy a template from one of the exercises and make relevant adjustments like changing the name of the topic to connect the template with our simulation.

Now, the next task would be to adapt the current template to our Evolutionary Robotics requirements.

## Problems and their Solutions
This week's work was quite interesting and something a little above my skills, allowing me to acheive the [flow state](https://en.wikipedia.org/wiki/Flow_(psychology)). However, here are some lessons learnt:

- Gazebo has a **model editor** option. The model editor allows us to make changes to our robot models visually. We can add links and joints to our robot model and then save them as required.

- ROS allows us to package a number of sensors of the same type into a single topic. The node that subscribes to the topic can distinguish between the various sensor readings by means of the header.

- I haven't seen any simulation of the Line Follower robot which works by means of IR sensors on Gazebo. The IR sensors detect a black path on the ground based on it's reflectance property. This made me to believe that robots based on such sensors are not possible to simulate on Gazebo. However, according to the things learnt in this week, it seems that it is quite possible, but it may be a little involved and difficult to do so!

- Just a curious point though, why don't the Open Source organizations provide their installation instructions in the form of bash scripts. For instance, to [download ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) we have to follow a number of steps, instead the creators would have provided a bash script to automate the process. The same goes with the download of Robotics Academy [General Infrastructure](http://jderobot.github.io/RoboticsAcademy/installation/#generic-infrastructure).





