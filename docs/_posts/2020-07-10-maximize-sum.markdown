---
layout: post
title:  "Week #5: Maximize Sum of Numbers"
date:   2020-07-03 10:00:00 +0530
categories: genetic-algorithm
comments: true
---
Maximizing a sum of numbers chosen randomly from 0 to 1 using a simple Genetic Algorithm, was the major task of this week!

## Where does this fit in the project?
A Genetic Algorithm library is very crucial to our project. Tuning the parameters of the Neural Network, in essence, the intelligence of the robot is carried out through the genetic algorithm. The student should be able to adjust the population size, number of generations, probability of mutation and **the fitness function** of the algorithm which will form the solution of the exercise we are developing.

## Outcome of this Week
There is a genetic algorithm class by the name of `GeneticAlgorithm()` in the code developed for this week, that takes in the adjustable parameters from the user. Running the algorithm then randomly directs the chromosomes to maximize their fitness. This is the simplest version of the genetic algorithm without Neural Networks and Gazebo. Along with it a small example demonstrating the use of the genetic algorithm API was also developed. The example shows the adjustment of all possible parameters of the algorithm and tries to maximize the sum of the numbers that are present in it's chromosomes.

Along with it, a feature to save the logs of the algorithm during run-time was also developed. The logs keep track of the evolution of the solution(how fitness values changed with generations), the best chromosomes of each generation and 75%(adjustable quantity) of the complete generation(all the chromosomes of each generation). These features would allow the student to replay the code as and how they require!

Linking this algorithm to the Neural Networks would be the task of the next week, as we get closer and closer to the simulation of the first robot using Neural Networks and GA!

## Logic of the Code
As expected, the code involved extensive usage of numpy and it's random sublibrary. The entire code can be broken down into 5 main parts:

- **Random Generation**: The first step of any Genetic Algorithm is the generation of a population with random values of alleles. The algorithm starts it's optimization from this random generation. The random library of numpy helps generate a matrix of population with the required dimensions.

- **Fitness Calculation**: The most important step of this part is carried out by the user; which is, determining the fitness function. Most of this part involves the usage of loops that calculate the fitness for each individual of the generation.

- **Selection**: Roullete based selection method is used to select the individuals according to a probability distribution. The probability distribution is determined by normalizing the fitness values to lie between 0 and 1 and sum up to 1. This is accomplished using `numpy.random.choice` method that enables choosing elements from an array according to a probability distribution.

- **Crossover**: Single point crossover algorithm is accomplished using List Slicing.

- **Mutation**: Randomly mutating the alleles according to probability of mutation is done using `numpy.random` methods.

## Problems and their Solutions
There were some major problems and curiosities about the overall code this week.

- **Parallel Computing**: Genetic Algorithms are required to calculate the fitness values of each individual of each generation. This calculation should theoretically be sped up by the use of parallel computation. In practical terms, I tried using the `multiprocessing` and `threading` libraries of Python to perform the computation. And to my surprise(everyone's I guess), they did not speed up the computation even a little. Instead the algorithm started working slower than the sequential version. Whatever the reason is, it is for sure that after making simulations work with GAs, applying these concepts of parallelization may even be more difficult.

- **Widgets**: Robotics Academy exercises use templates along with certain widgets that help the user debug and code their solution. Soon, these GA exercises would also need to work with templates such as those. The next most creative problem would be to come up with a dynamic widget that allows the students to debug their code while solving these Genetic Algorithm exercises.

![Failed Parallelization](https://i.redd.it/9tu18n684z331.jpg)

*Current architecture*

