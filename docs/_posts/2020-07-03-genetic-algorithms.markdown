---
layout: post
title:  "Week #5: Genetic Algorithm"
date:   2020-07-03 9:45:00 +0530
categories: genetic-algorithm
comments: true
---
This week’s task involved reading and getting familiar with the concepts of Genetic Algorithms. Thinking about their implementation was the main task.

## Where does this fit in the project?
The next task in the project, after the development of Neural Network library is the development of the Genetic Algorithm template. For a particular exercise, the student will be asked to design a fitness function and test whether the robot learns the right behaviour or not! The development of Genetic Algorithm template will allow the student to accomplish this. By default, a space will be left for the student to write his/her code in the template, which will serve as the fitness function for the genetic algorithm.

## Outcome of this Week
A structured and well defined path towards the generation of the Genetic Algorithm was developed in this week. How genetic algorithms function, what components are they made up of and their implementation was researched and studied. Here is the material referenced:

- These slides on [Algoritmos Genéticos (Genetic Algorithm)](http://www.robolabo.etsit.upm.es/asignaturas/irin/transparencias/AG.pdf) and [Robótica Evolutiva(Evolutionary Robotics)](http://www.robolabo.etsit.upm.es/asignaturas/irin/transparencias/ER.pdf) by Álvaro Sir

- [This video](http://www.robolabo.etsit.upm.es/asignaturas/irin/transparencias/IRIN_20200417.mp4) demonstrating the IRSIM simulator by Álvaro Sir

- [This paper](http://www.robolabo.etsit.upm.es/asignaturas/irin/papers/floreano.sab94.pdf) on one of the first evolutionary robotics experiment carried out by D. Floreano and F. Mondada

## Logic of the Code
Not much code for this week! 

![Sharpen the axe](https://i.pinimg.com/originals/47/b3/71/47b371ec19a7d762136239ef605e24f4.jpg)

*A quote for every occasion*

This week was filled with theoretical knowledge. However, a small part involving code was the research on how various other people implemented their version of genetic algorithm. A very good [code is provided by ahmedfgad](https://github.com/ahmedfgad/GeneticAlgorithmPython). 

Overall, the logic of the code doesn't seem to be a complex part. Simple array manipulations and the use of random library(*Library used for generating random numbers, not any random library!*), along with some basic State Machine concepts are all that would be needed. The difficult part would be the integration of the code with Gazebo. However, next week would tell how easy or difficult something would be, when the actual work would be under progress!

## Problems and their Solutions
No major problems only some lessons:

- Genetic Algorithms are not any simple random based search, instead they are a **directed version of random search**. The randomness combined with a direction, make Genetic Algorithms quite robust. This property makes sure that it is difficult to find **GA Hard problems** in Computer Science. Which is a really **cool** thing I guess.

- Genetic Algorithms are quite difficult to study mathematically. The closest anyone can get to GAs mathematically is through the use of **Schemata Theorems**. However, even these theorems are filled with probabilities and are not suitable to study for GAs involved in some complex tasks, such as controlling a robot or building trusses.

- Just a side knowledge though, *mui* translates to very, *cero* translates to zero and *uno* translates to one from Spanish to English!



