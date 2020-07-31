---
layout: post
title:  "Week #8: Speed Learning"
date:   2020-07-31 10:00:00 +0530
categories: exercises
comments: true
---
As was mentioned in the previous week, the training part of the exercise was extremely slow! Various strategies were adopted this week to speed up the training.

## Where does this fit in the project?
The first exercise of the project is almost complete, BUT we have a little problem(little but tough to solve) that the training part of the exercise is extremely slow. It runs at about 1 individual in 2 seconds, but we need to get it work at about 50 individuals in 2 seconds. Considering the use of Python, a rate of 50 individuals in about 10-15 seconds is also enough. This speed up is an extremely important part of the project as this determines the use and success of it! Nobody would like to run an exercise that takes hours to learn even a simple task like obstacle avoidance.

So, speeding up the training part of the exercise is extremely important!

## Outcome of this Week
![Training-1](./../assets/gif/training-1.gif)

*Robot learnt something not right*

![Training-2](./../assets/gif/training-2.gif)

*Robot learnt nothing*

The outcome of this week is not that impressive, but a lot of different strategies were tested, thought out and applied to speed up the computations. After all the improvements, the current architecture of the exercise is as follows:

- **Tensorflow is not a part of the requirments anymore**. Tensorflow was the main bottleneck of the code(reasons described in the next section) and was redundant. The algorithm of the Artificial Neural Network library was strong enough that it did not require Tensorflow. Therefore, it was removed!

- The training part now runs **5 Gazebo instances in parallel**, and all of them headless. The testing part runs a single Gazebo instance without the headless mode. The implementation of this part was really interesting.

- The GUI Debug window is also improved, with some modification of the buttons and an improved way to select which generation to test or continue from!

Some honourable mentions of the additions that did not take place:

- Multiple robots in a single instance of Gazebo

- A seperate PyQt based simulator to train the robot

- Speeding up Gazebo, so that the simulation runs in a fast forward manner

All of the above improvements did speed up the code as required. But, another challenge arose with this. The **robot doesn't seem to learn any obstacle avoidance behaviour** at all. More on this in the last section!

## Logic of the Code
This week involved more of analyzing, deletions and restructuring of the code.

- Identifying that Tensorflow was taking up most of the time was easy. Just timing the code using `datetime` library allowed me to determine the problem. Removing Tensorflow was easier than that. The algorithm that I had implemented before was enough to determine the correct order of execution and Tensorflow was really just a redundant part. Although the removal of Tensorflow added a lot of loops, but the code was now faster than before.

- This [post on Genetic Algorithms](http://jaredmmoore.com/rosgazebo-genetic-algorithms-and-multiple-instances/) by Jared M Moore helped in setting up the 5 parallel instances of Gazebo. The trick behind this is to provide a seperate group namespace to each of the instance and allocate seperate ports. Other than that the clock part of Gazebo is also remapped to seperate namespace so that every instance runs completely independent of each other.

- Some simple modifications to the code allowed me to make the relevant changes in the GUI Debugger window. 

## Problems and their Solutions
The problems of this week were:

- **Not learning obstacle avoidance behaviour**: This was the most frustrating problem of the week. I thought(and as suggested by Álvaro Sir) the major reason behind this was a small evaluation time. But, even increasing the evaluation time did not result in a better performance. Now, this points to a flaw in my Genetic Algorithm library, or the evaluation time is still less. Whatever the case is, it's really important to rectify this!

- **Deciding on the improvement**: Not much of a problem though, but I had to try a lot of different things before coming up with the solution of running parallel Gazebo instances(still not sure, if this is the correct solution) like trying to speed up the simulation time of Gazebo and multiple robot instances in a single simulation. Nonetheless, searching for a solution was an extremely fun activity.

![Edison Quote](https://www.azquotes.com/picture-quotes/quote-i-will-not-say-i-failed-1000-times-i-will-say-that-i-discovered-there-are-1000-ways-thomas-a-edison-93-54-27.jpg)
*A quote for every occasion*





