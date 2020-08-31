# Obstacle Avoidance

## Goal
The goal of this exercise is to make the robot learn obstacle avoidance behaviour by means of a Neural Network adjusting it's parameters using Genetic Algorithms

## Installation
Follow the instruction in this [README](./../README.md)

## Running the Exercise
Navigate to the obstacle avoidance exercise directory

```bash
cd colab-gsoc2020-Sakshay_Mahna/evolutionary_robotics/obstacle_avoidance
```

The current working directory will be changed.

### Robot and it's applications

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
## Theory
Coming Soon!

## Hints
Coming Soon!
