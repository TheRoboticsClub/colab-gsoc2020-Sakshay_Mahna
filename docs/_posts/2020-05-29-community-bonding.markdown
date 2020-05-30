---
layout: post
title:  "Community Bonding"
date:   2020-05-29 10:00:00 +0530
categories: community-bonding
comments: true
---
Google Summer of Code starts with the Community Bonding period where we get to know about the community and get familiar with the code base and work style.

For me, this month was packed full of new knowledge and experience. Learning about Evolutionary Robotics and Neural Networks was amazing! It was really difficult to resist the urge to code all the new knowledge that I had acquired in this month. Nonetheless, the coding period is now just 3 days away!

## Evolutionary Robotics
Evolutionary Robotics is a set of methods that uses **principles of evolution** to develop controllers/logic/hardware for autonomous robots. By taking *populations* of candidates we are able to select the best architecture/design/values of the controllers that we desire. The **fitness function** plays the most crucial role in our mission. A well designed fitness function determines the task and performance that the robot is able to learn.

In the exercises we will develop, our main objective is to make the student design the fitness function, according to the exercise.

## Neural Networks
In a biological sense, neural networks are a collection of neurons. Simulating the same behaviour on a computer, to perform intelligent tasks like classification, pattern recognition are called **Artificial Neural Networks**. ANN can also be refered to as a computational graph, where each node in the graph performs a computation on the input and sends it to the next node along the path. Hence, just by varying the connections of the graph or the number of nodes, we can get a **huge variation** in the architecture of the Neural Network.

In our context, we will provide an API to the student, through which the student can select/design the architecture of an ANN, based on it's connections, number of layers and number of nodes in each layer.

### Deep Learning v/s Artificial Neural Networks
For a student like me, the above title would seem incorrect, as deep learning implements neural networks behind the scenes, so how come a v/s in between. Let's try to justify that!

A simple Google/Youtube search(leaving some exceptions) on artificial neural networks will yield results on deep learning. Well this is good for someone starting out on ANNs but for someone who wants to delve into the concepts of neural networks this is quite bad. Deep Learning nowadays have taken the stage and hid all the concepts that go in the development of a sophisticated Neural Network.

Neural Networks is such a wonderful topic, that has many concepts to learn. For instance, the following is the list(not exhaustive):

- Neural Networks employ a number of learning mechanisms, like **Hebbian, Competitive or Boltzmann**. Each of the mechanism has a particularly well defined **update equation** which has their own advantages and disadvantages.

- **Gradient Descent** is *not* the only optimization method used by Neural Networks. There exist many other methods as well such as the **Newton method** and the **Gauss-Newton method**

- The **Radial Basis Function** networks intuitively transform the input vector into a higher dimensional space, where the classification task becomes easier and then map it to output according to our specifications. For this purpose, **Green's functions** are employed by the RBF networks.

- **Self Organizing Maps** are a method to employ unsupervised learning in Neural Networks.

From all the experience, I beleive that Deep Learning and Neural Networks should be treated seperately, even on internet discussions! 

## Resources
The resources shared by my mentors were really amazing which helped me learn all the concepts required. Following is the list:

- [Evolutionary Robotics](https://mitpress.mit.edu/books/evolutionary-robotics) by David Floreano and Stefano Nolfi. Great for our purpose as it includes a collection of experiments in this field.

- [Evolutionary Robotics: From Algorithms to Implementations](https://books.google.co.in/books/about/Evolutionary_Robotics.html?id=xyqpfWOjpdMC&source=kp_book_description&redir_esc=y). This book again consists of experiments, and also includes some details on Fuzzy Logic controllers. The last chapter on making robots move through Genetic Algorithms is covered in much detail.

- This [website](http://www.robolabo.etsit.upm.es/subjects.php?subj=irin&tab=tab3&lang=es) consisting of the study material provided by Álvaro Sir. The material was in Spanish, which provided a very good opportunity to learn some basic spanish. Ke buena to be a part of GSoC!

- This [amazing course](https://nptel.ac.in/courses/117/105/117105084/) by Prof. Somnath Sengupta. Includes lot of mathematics, but covers Static Neural Networks really good!

![Fleunt in Google Translate](https://qph.fs.quoracdn.net/main-qimg-ff3081cff8e19090fa6e6d83cd330855)

*Quite Relatable!*

## Conclusions
The coding period will start on the coming Monday. It was really amazing learning all the new things in this Community Bonding Period, and I really can't wait to apply all of them!
