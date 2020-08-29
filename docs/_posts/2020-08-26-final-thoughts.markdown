---
layout: post
title:  "Week #12: Final Thoughts"
date:   2020-08-26 9:40:00 +0530
categories: exercises
comments: true
---
Finally, we are approaching the end of GSoC. This blog post describes some final thoughts and comments regarding the future scope and improvements to the project.

## Achievments
During the course of these 3 months, we developed the first working Artificial Intelligence exercise of JdeRobot Robotics Academy. The contents of the project include:

- **Neural Network Library**: An easy to use and simple Neural Network library that helps students develop Static or Dynamic Neural Networks.

- **Genetic Algorithm Library**: This library can be used to apply Genetic Algorithm to common optimization problems.

- **Obstacle Avoidance Exercise**: An exercise, where students need to enter an appropriate fitness function to train the robot for a certain behaviour(obstacle avoidance in our case).

![Good Behaviour](./../assets/gif/good.gif)

*A well trained robot*

## Future Scope
We weren't quite able to accomplish all the objectives of the proposal, but completed all we could. The current exercise itself has a lot of scope for improvement. Some of the improvments future developers(or me) can focus on are:

- **Speeding up the process**: The current obstacle avoidance exercise takes 2 hours to train, which is acceptable for this exercise. However, scaling this to other exercises will be difficult as they would take even longer time.

- **Determinism**: Deterministic environment is important to help students debug and understand their mistakes. The current software is not deterministic, which also does not enable us to use elites.

- **Collision**: After experiencing a collision, the robot model behaves in an errotic manner. This records errors in the sensor input and motor output, influencing the fitness value. This collision should be avoided or removed.

All of the above issues are mostly part of training. However, all of these problems could be solved(just a hypothesis) using a minimalistic Physics Engine, which can be used to train the robot. The robot can then be tested on the real Gazebo simulator.

## Further work
The other 4 branches contain some of the work that was left in progress, or did not make it through due to some constraints:

- **master**: The main working branch.

- **genetic_engine**: This contains some starting work of the parallel physics engine for the exercise.

- **step_plugin**: This contains some attempts at making the simulator deterministic by controlling each and every time step of the simulator.

- **tensorflow**: This contains the tensorflow implementation of Neural Network library. However, this could not be included in the final project, as this made the training process even slower. Simple use of numpy library was enough to get the library working.

- **update**: Contains some final touches to the documentation and code.

## Personal Thoughts
The 3 month journey of GSoC was quite amazing. I really enjoyed writing the code and learning new things about coding, neural networks and evolutionary robotics. It was great being mentored by Álvaro Sir and Luis Roberto Sir.

I would like to **work further on the project** for one month more and try out the possiblity of the parallel Physics Engine approach. Seeing, a perfect exercise that can be used by students would be really great to experience!

## Lessons learnt

1. Before starting work on a project, be sure to check out it's scope whether it is possible or not. If we had checked this repeatbility part of Gazebo before starting to work, we could have come up with another way to train the robots. However, I am not completely sure whether the exact problem was with Gazebo or my code. I would be really satisfied, if the mistake is in my code, as this would make the exercise complete in itself, just using ROS and Gazebo similar to other Robotics Academy exercises.

2. Simple and easy to use interface is very important for our users. Even the slightest of details should be mentioned in our manual or should be understandable by the context. All of this is very easy to code and create, we just have to be mindful enough to include these things!

3. A lot of new Python tricks and concepts: Logging, EAFP, Duck Programming, Documentation styles and Unit Testing.
