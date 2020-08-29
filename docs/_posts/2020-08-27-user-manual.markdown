---
layout: post
title:  USER MANUAL
date:   2020-08-27 10:40:00 +0530
categories: exercises
comments: true
---
The User Manual of the software developed in the GSoC program.

![Teaser](./../assets/img/teaser.png)

*Obstacle Avoidance Exercise*

The link to the [Github Repository](https://github.com/TheRoboticsClub/colab-gsoc2020-Sakshay_Mahna)

# Installation
The following installs the software on your system.

## Prerequisites
The libraries and exercises are developed and tested in **Python 2.7.17, Pip 20.0.2, ROS Melodic, Ubuntu 18.04**.

### Git
The instructions to install Git(command terminal) for Ubuntu 18.04 are:

- Update the Default Packages

	```bash
	sudo apt update
	```

- Install Git

	```bash
	sudo apt install git
	```

- Check if the following command does not give a missing error

	```bash
	git
	```

![Git works successfully](./../assets/img/git_command.png)

*Successful installation*

### Python and Pip
The instructions to install Python2.7 for Ubuntu 18.04 are:

- Update and Upgrade the Default Packages

	```bash
	sudo apt update
	sudo apt upgrade
	```

- Install Python2.7

	```bash
	sudo apt install python2.7
	```

- To check correct installation, the following command should open a Python interpreter

	```bash
	python2
	```

- Install Pip for Python2

	```bash
	sudo apt install python-pip
	```

- Check if the following command does not give a missing error

	```bash
	pip
	```

![Python works succesfully](./../assets/img/python_command.png)

*Successful Installation*

### Generic Infrastructure of Robotics Academy
Follow the Installation instructions of **Generic Infrastructre of Robotics Academy** as given on the [Robotics Academy webpage](http://jderobot.github.io/RoboticsAcademy/installation/#generic-infrastructure).

The installation is done correctly if we can successfully run the following commands:

- Source the environment variables

	```bash
	source ~/.bashrc
	```

- Start the ROS Master server. This would keep running in the terminal without giving any errors

	```bash
	roscore
	```

- The Gazebo Model variable should contain paths to jderobot directories

	```bash
	echo $GAZEBO_MODEL_PATH
	```

![ROS installed perfectly](./../assets/img/ros_command.png)

*Succesfull installation*

## Dependencies
The project uses the following python modules

```
numpy==1.16.5
graphviz==0.14
matplotlib==2.1.0
```

These dependencies can be downloaded seperately or through the commands given ahead.

## Libraries
The libraries developed for the project are available in [libraries](https://github.com/TheRoboticsClub/colab-gsoc2020-Sakshay_Mahna/tree/master/evolutionary_robots/libraries). These libraries are useful for solving the exercises. The API reference and examples are also provided.

## Installation
Before running the installation, make sure that all the prerequisites are already installed on the system which are **Git, Python, Pip and Generic Infrastructure of Robotics Academy.**

- Open a new terminal and navigate to the directory where the exercises should be downloaded.

- Clone the Github Repository.

	```bash
	git clone https://github.com/TheRoboticsClub/colab-gsoc2020-Sakshay_Mahna
	```

- Navigate to the working directory inside the cloned repository.

	```bash
	cd colab-gsoc2020-Sakshay_Mahna/evolutionary_robotics
	```

- Update Pip to the latest version. Some dependencies need the latest version to install correctly.

	```bash
	pip install --upgrade pip
	```

- Install the dependencies. All the dependencies would be installed without giving any errors.

	```bash
	pip install -r requirements.txt
	```

- Source the ROS environment variables.

	```bash
	source /opt/ros/melodic/setup.bash
	```

- Run the installation script to configure the Gazebo Assets. This will build the workspace and make new directories `devel` and `build` inside `colab-gsoc2020-Sakshay_Mahna/catkin_ws` directory.

	```bash
	. installation.bash
	```

- Run the source script to source the Gazebo Assets. This command would add new paths to `GAZEBO_MODEL_PATH` environment variable.

	```bash
	. source.bash
	```

The application has now been installed.

## Running the Exercise
Navigate to the obstacle avoidance exercise directory

```bash
cd colab-gsoc2020-Sakshay_Mahna/evolutionary_robotics/obstacle_avoidance
```

The current working directory will be changed.

### Robot and it's applications

![Roomba](./../assets/img/robot.png)

*Your Robot for this exercise*

The robot used for this exercise is the Roomba. It has the following features:

- Differential Drive, there are 2 motors that control the motion of the robot.

- There are 8 infrared sensors around the body of the robot.

### How to perform the exercise?
The student has to edit 3 different sections in `MyAlgorithm.py` file:

- Enter the various parameters for the Genetic Algorithm

  - `POPULATION_SIZE`: The number of individuals in a generation.
  	```python
  	POPULATION_SIZE = 
  	```
  - `NUMBER_OF_GENERATIONS`: The number of generations to train the robot.
  	```python
  	NUMBER_OF_GENERATIONS = 
  	```
  - `MUTATION_PROBABILITY`: The probability by which a gene of the chromosome will be mutated(Randomly changed).
  	```python
  	MUTATION_PROBABILITY = 
  	```
  - `EVALUATION_STEPS`: The number of time steps for which each individual is going to be evaluated.
  	```python
  	EVALUATION_STEPS = 
  	```

- Enter the fitness function in the function `fitness_function`. The function should return (integer/float) value of fitness. The following section [Application Programming Interface](#application-programming-interface) describes each of the parameters of the function.

	```python
	def fitness_function(left_motor_speed, right_motor_speed, infrared):
		# Code the fitness function here
		fitness = 0
		
		return fitness
		
	```

- Enter the neural network specifications in the function `define_neural_network`. The input and output layers have to be defined with the given parameters. The student may add additional hidden layers. **Remark**: Since, the robot we are using has 8 infrared sensors, therefore, our `inputLayer` has 8 neurons with `IdentityActivation()` and `INFRARED` sensor input. The `outputLayer` has 2 neurons(2 motors) with `TanhActivation()` and outputs to `MOTORS`.

	```python
	def define_neural_network():
		# Define the layers
		# Layer(name_of_layer, number_of_neurons, activation_function, sensor_inputs, list_of_output_layer_names)
		inputLayer = Layer("inputLayer", 8, IdentityActivation(), "INFRARED", ["outputLayer"])
		outputLayer = Layer("outputLayer", 2, TanhActivation(), "", ["MOTORS"])
		
		# Define the neural network
		neural_network = ArtificialNeuralNetwork([inputLayer, outputLayer], "STATIC")
		
		return neural_network
	```

Altering the code here influences the behaviour of our robot.

#### Application Programming Interface

- For the specifics of setting the Neural Network, refer to this [API reference](https://github.com/TheRoboticsClub/colab-gsoc2020-Sakshay_Mahna/blob/master/evolutionary_robots/libraries/neural_networks/README.md), the API required to code the fitness function is discussed ahead

- `left_motor_speed`: The speed of the left motor wheel of robot

- `right_motor_speed`: The speed of the right motor wheel of robot

- `infrared`: List of 8 infrared sensor values


### How to run your solution?

- Source the source script in this directory. This sources the various environment variables, just to avoid any problems!

	```bash
	. source.bash
	```

- Launch the Gazebo simulation in the same terminal window. Ignore the yellow colored warnings that appear in the terminal. **We should run always run the train file of the simulator, when we want to train the algorithm and test version when we want to test**.

	```bash
	# Complete Simulation, only during testing
	roslaunch ./launch/test.launch

	# Headless version, only during training
	roslaunch ./launch/train.launch
	```

- In a new terminal window, execute the academic application that will incorporate your code based on whether we want to train or test the robot. This would open a GUI application through which the user can execute the code.

	```bash
	# Only during training
	python2 ./train.py

	# Only during testing
	python2 ./test.py
	```

- The training GUI has 2 buttons. The first button `Start Training`, starts the training of the exercise from scratch. All the previous logs are deleted if this button is clicked. The second button `Resume Generation` works with the input box present alongside. It resumes the training from the specified generation, **if the log file of that generation is present**. Typically, generation numbers with multiples of 25 and the generation at which the user closed the training are available. **Due to a bug, a button when clicked cannot be deselected. In order to select any other button, please close the GUI application, open it again and then select the button**.

- The test GUI has a single button. `Test Best Chromosome` tests the best chromosome that was found in the previous trainings. It also works with the input box.

![Succesfull Training](./../assets/gif/train_command.gif)

*Training Illustration*

![Succesfull Continuation](./../assets/gif/continue_command.gif)

*Resume Illustration*

![Successful Testing](./../assets/gif/test_command.gif)

*Test Illustration*



