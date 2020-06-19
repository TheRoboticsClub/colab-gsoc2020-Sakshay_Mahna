---
layout: post
title:  "Week #3: Morphology of Neural Network"
date:   2020-06-19 10:00:00 +0530
categories: neural-network
comments: true
---
This week's task involved modifying the architecture of Neural Network template. The aim was to develop a more scalable and modular template, along with an easier user interface.

## Where does this fit in the project?
For most of the evolutionary robotics experiments, Neural Networks play a major role in deciding the control and intelligence(decision making process) of a robot. Particularly, for the exercises we are developing, Neural Networks are the **brain** of the mobile robots we are going to use. The various weights, gains and biases are to be tuned by means of a genetic algorithm, which will enhance the control and intelligence of our robot.

For the exercise, the student should focus on coding the genetic algorithm, rather than the structure and definition of the Neural Network. In order to make the selection of Neural Network easier, we are going to provide a template to the student. Through which, the student can work on the genetic algorithm, and pay less attention to the Neural Network.

Creating such a modular and scalable template was the aim of this week.

## Outcome of this Week
The library is now very easy to manage and use. Everything is systematic in terms of the input and output of the network. The Neural Networks can now potentially work on 4 categories of layers, which are the Input Layer, Hidden Layer, Associative Layer and the Output Layer. The user can decide upon which type of layer to chose be it Static or Continuous. Intelligent Exceptions are now raised on wrong user input and the use of *dictionaries* in various places allows easier accessibility and understanding to the user.

## Logic of the Code
This week was really intense in terms of the logic. Although, Simple Neural Networks were developed earlier, but they **lacked some simple qualities like scalability and understanding**.

The first step towards this goal was the understanding of some examples of the Neural Network that the student was going to require. [Here is the material](http://www.robolabo.etsit.upm.es/asignaturas/irin/transparencias/ER.pdf) that led me through it. Now, comes the logic!

Neural Networks are more or less a directed computational graph. The networks perform a series of sequential operations on some vector input and generate a vector output based on some weights and biases. To make their creation more logical and easier to understand, we provide the user with a Layer interface(**Documentation part is left, I'll complete by today!**). The Layer has attributes like number of neurons, type of layer, activation function, input connections and output connections. By defining these for each layer, we later add them to the Artificial Neural Network object.

Once the order of execution is decided, the network uses a dynamic programming algorithm to save the outputs in the order of execution. Based on those saved outputs, the network generates the output based on the input. 

![Steve Jobs Creative Design](https://www.geckoandfly.com/wp-content/uploads/2014/02/steve-jobs-quotes-creative-apple-design-09.jpg)

*Something I learnt along the way*

## Problems and their Solutions
This week was really intense on the logic. Therefore, there were quite a lot of problems to be faced. Here are the 2 major ones:

- **Easier User Interface**: It was really difficult to understand, how much simplicity would be required by the user of our software. Most of the time I ended up providing an interface that had too much complexity. The interface appeared simple in my thoughts, but the reality of situation arose on thinking about the use-case of a user. But, my mentors helped me in facing this problem. The idea of directed graphs and a layer interface were thought of, by them.

- **Theory to Practice**: Before, working on this problem, I had only read and studied about directed graphs and other related algorithms. Thinking about how they were working in our case, and developing an algorithm that operated according to the order of execution was really fun!


