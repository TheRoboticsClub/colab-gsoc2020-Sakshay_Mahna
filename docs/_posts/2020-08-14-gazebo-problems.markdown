---
layout: post
title:  "Week #9: Problems with Gazebo"
date:   2020-08-14 13:30:00 +0530
categories: exercises
comments: true
---
This blog post mainly describes my experience with Gazebo for this week.

## Where does this fit in the project?
So building up from previous posts, we are to satisfy the following conditions for any good Genetic Algorithm based exercise:

- Fast enough training time(since the algorithm is highly parallelizable)
- Deterministic and Repeatable simulations

The current development of the exercise, somehow satisfies the first condition, but the second condition is not satisfied. So, this week was spent on trying to make the simulations repeatable.

## Outcome of this Week
After a lot of reading, asking for help and trying out Github projects of various people in the field, I finally came to the conclusion that this is not possible in Gazebo(reasons in the next section). Therefore, we need to focus on some different process of training. Since, I discovered some new ways to train, this week has some positive outcomes and scopes for the future.

![Non deterministic](./../assets/gif/nd.gif)

*Not only the time to reach destination, even the sensor readings are different*

## Logic of the Code
The logic can be divided into 3 parts:

- **Attempts to make Gazebo deterministic**: Some of the attempts I made to make Gazebo simulations repeatable was by tuning the `rate` and `queue_size` of Publishing motor messages, applying latches to the Publisher. This [blog post](http://jaredmmoore.com/) by Dr. Jared M Moore was really amazing and helpful(unsuccessful however). He has worked on some projects involving Evolutionary Robotics on ROS and has a lot of Github repositories for the same. I tried replicating the settings and configurations, but still the results were the same undeterministic simulations. Apart from the technical part, I also asked for help from JdeRobot developers(Pankhuri Vanjani, Nacho Condés and JoseMaria Sir provided some hints), but still there was no progress.

- **Realization of non-deterministic simulation**: It took quite some while to reach to the conclusion, that making such simulations which are repeatble is very difficult(I wouldn't say impossible) atleast in the time that I have left(2-3 weeks). Some of the supporting answers that I found on GazeboSim answers, [answer1](https://answers.ros.org/question/11052/is-the-gazebo-simulator-deterministic/) and [answer2](https://answers.gazebosim.org//question/25010/repeatability-of-experiments-and-determinism-of-gazebo-simulation/). The problem with ROS Gazebo is that since they are different software, ROS provides a way to communicate with Gazebo asynchronously. Due to this, the messages that we send are sometimes displaced in time, which adds the randomness part to the simulation. In order to acheive deterministic environment, we need control of each and every state of the simulation. Hence, they should be on the same thread.

- **Alternatives**: To deal with this problem, the solution(which is yet to discuss in today's meeting) is to use seperate environment for training and testing. The best I could think of, is a minimal self designed Physics Engine. We can create a non visual Python based simulator. Normal simulators like Gazebo, provide us with a lot of options, most of them which we are not using at all. The simulation only requires a robot body, distance measurement from walls, linear and angular speed along with some minimal collision dynamics. All of this can simply be coded up in Python, without any visuals. This approach would be quite scalable(due to the use Classes like Sensor, Robot and Environment class), fast(all the members of a particular generation can be run on multiple threads at a single time) and completely deterministic, as we would have a full control over the states. The training phase only has to generate data about the function that maps sensor values to motor outputs.

![Change of Plans](https://izeyodiase.com/wp-content/uploads/2017/01/If-the-plan-doesn%E2%80%99t-work-change-the-plan-not-the-goal.-1024x1024.jpg)

*If Gazebo can't do it, Python will do it*

## Problems and their Solutions
The problems of this week were:

- **Dealing with dissatisfaction**: There were a lot of unsuccessful experiments in this week. Every other alternative didn't seem to work. Nonetheless, I learnt a lot about Gazebo Plugins and internal Gazebo services. It also provided me with an oppurtunity to open up and ask for help from various people.

- **Patience**: This week was a real test of my patience. It also helped me discover the limits of my patience. How long can I go through failed attempts.





