---
layout: post
title:  "Week #2: Documentation"
date:   2020-06-12 10:00:00 +0530
categories: documentation
comments: true
---
This week's task involved writing the documentation of the library coded last week.

## Where does this fit in the project?
What good is a library if people don't know how to use it! Documentation forms a really important part of a software or any technology. The users of a library read the documentation to understand how to use it in their project. The Neural Networks library developed previously is really extensive, hence a proper documentation is needed to easily use the library. Apart from it, the comments in the source code file are also really important. They allow other developers and our future selves to understand what were the design decisions that we took, and how they were implemented. Hence, documentation and commenting are the most important part of this software.

![Explain Quote](https://thequotes.in/wp-content/uploads/2016/05/Albert-Einstein-Quotes-4.jpg)

*Related Quote*

## Outcome of this Week
The documentation and commenting of the source code is completely up to date now. The documentation explains the following:

- How to build a simple Neural Network?
- How to use Activation Functions?
- How to use the different Neural Networks?

Apart from it, the Numpy style docstring documentation is also provided for the source code. The docstring clearly explains the attributes and methods present in a class and the input/output parameters of a function of that class. From now on, calling the python function `help()` with any library class or function will provide it's complete documentation about what to expect from it!

## Logic of the Code
There is one very important distinction between comments and documentation. Documentation is for the users of the library, who want to use the library functions and get them to work for their project. These users **do not need to know about the workings of the code**. However, comments are provided in an application for the developers that explain the working and logic behind each line of code. A developer **needs to know about the workings of the code**. Taking these differences into account, documentation and comments need to include details that are relevant to **their** users.

Docstrings are also provided in the Python code to explain the general purpose of each class and function. There are various styles of docstrings for Python. These are Numpy, PyDoc, EpyDoc and many more. **Sphinx** is a documentation tool, that generates seperate html documentation files from the Numpy docstring format.

We also had the choice of using the Sphinx generated documentation or the simple README based documentation. I chose the README based documentation because of it's readability and clear representation. The Sphinx documentation, although is great for presentation and would have seperated the the API and exercise documentation very easily. But, looking at the generated documentation, it appeared to be quite cluttered and not that clear as a README file looked. Therefore, I chose the README documentation.

## Problems and their Solutions
Overall, this week was really amazing! Documentation is one thing that I have not really worked upon in my past projects. So, this was a new thing for me. Learning about the differences between comments and documentation, the various docstring formats and the auto generated documentation was a really great learning experience.


