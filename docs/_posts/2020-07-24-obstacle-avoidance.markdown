---
layout: post
title:  "Week #7: Obstacle Avoidance"
date:   2020-07-24 10:15:00 +0530
categories: exercises
comments: true
---
This week’s task involved working on the backend of the exercise. The first exercise starts to take shape in this week. 

## Where does this fit in the project?
The first exercise of the project is Obstacle Avoidance(I think I have described this a various times before, nontheless, here we go again!). In this exercise, the student is supposed to design a fitness function for a Genetic Algorithm that adjusts the parameters of the Neural Network. Hence, the user has the degree of freedom(I've been studying about Manipulators a lot recently!) to design the Neural Network, adjust the parameters of the Genetic Algorithm and design the fitness function.

To make the exercise Robotics Academy ready, it also needs to have a GUI Debugger. The debugger is used to run the exercise, stop the exercise and measure the various sensor values and debugging information. The Gazebo simulation developed in the previous week, the backend and GUI developed in this week, constitute the entire exercise.

## Outcome of this Week
The backend of the exericse is complete. The `MyAlgorithm.py` file provides the student with a template, where he/she can fill in the required code. Spaces have been left for the Neural Network, fitness function and the parameters of the Genetic Algorithm. Along with this the debugger GUI of the exercise is also ready.

![Training](./../assets/gif/training.gif)

*Running the exercise*

All in all, I would say the exercise is complete and ready for student's use, except some modifications that would be suggested by my mentors in today's meeting!

## Logic of the Code
Interesting work again this week! 

- The first task was to design the GUI and it's relevant backend. Robotics Academy uses PyQt to render the GUI. PyQt provides a visual interface to place the various blocks and placeholders and name them accordingly. Once the interface was ready, it was converted to a Python file using the command `pyuic`. Another python file describing the interface of GUI with the user was developed in the `GUI.py` file. The GUI has 4 buttons: to train the robot, to resume training, to continue training from a portion and to test the best chromosome till now. Along with that, the GUI describes the various statistics during training.

- The above task was quite interesting to do! Due to that I had made the code very complicated and inefficient, which resulted in a slower training after some iterations(which is a case still now! Need to discuss this!). Therefore, the second task was to define some code in another file, called `GA.py` to keep the student template `MyAlgorithm.py` clean. Calculation of output from Neural Network was made efficient and some modifications were applied to `ann.py` to speed up TensorFlow. There are still some little problems here and there, which I will be working on, in the coming days!

## Problems and their Solutions
The more difficult the work, the more things you get to learn from the experience! Just a thought though! Here are the learnings of this week!

- **Tensorflow code**: Tensorflow works by means of communicating information(tensor) through nodes(flow). This provides us with a computation graph through which we compute the output to our desired inputs. There are various operations in Tensorflow, like `assign`, `add`, `numpy_function` and so on. Each operation adds a new node to the Tensorflow graph. This is a really big bottleneck, as more nodes are added, our code begins to slow down due to usage of RAM. This simply implies that the Tensorflow code in `ann.py` can be optimized further to speed up. However, a GPU can also be added, through which Tensorflow can speed up the calculations. Interesting point, **Software as well as the Hardware determine the time complexity of an algorithm**.

- **Motor Speeds**: This is a good problem. ROS does not provide individual motor speeds of the robot. We simply provide a linear(vx) and an angular(az) speed, which moves the **robot model** in the simulation. Therefore, there is no concept of motor speeds. And we require, individual motor speeds for the exercise. A trick to work around this was to identify, that sum of motor speeds is proportional to linear speed and the difference of motor speeds is proportional to angular speed. But, I am not really sure, this trick works good or not because I was not able to fully train my robot.(See the next point)

- **Training**: Me having an impatient personality, in general cannot wait long enough to see the robot train itself to the end. This added to the slowing down of training after some iterations makes it really difficult to wait for such long. Earlier, I oversaw the training of 3 generations with 10 individuals taking 300 time steps each. Let's put them to training again!





