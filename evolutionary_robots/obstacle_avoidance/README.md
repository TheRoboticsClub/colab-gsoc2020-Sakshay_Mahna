# Obstacle Avoidance

## Goal
The goal of this exercise is to make the robot learn obstacle avoidance behaviour by means of a Neural Network adjusting it's parameters using Genetic Algorithms

## Installation
Follow the instruction in this [README](./../README.md)

## How to run your solution?

- Source the environment variables

```bash
source /opt/jderobot/setup.bash
source /opt/jderobot/share/jderobot/gazebo/gazebo-setup.sh
source /opt/jderobot/share/jderobot/gazebo/assets-setup.sh
```

- Run the source script of this repository

```bash
cd ..
. source.bash
cd obstacle_avoidance
```

- Launch the Gazebo simulation in the same terminal window. A new window showing the Robot Model in an environment would appear. Ignore the yellow colored warnings that appear in the terminal.

```bash
roslaunch ./launch/obstacle_avoidance.launch
```

- In a new terminal window, execute the academic application that will incorporate your code. This would open a GUI application through which the user can execute the code.

```bash
python2 ./exercise.py
```

- The GUI has 4 buttons which provide 4 different ways to run the exercise. The first button `Start Training`, starts the training of the exercise from scratch. All the previous logs are deleted if this button is clicked. The second button `Continue Training` resumes training from the generation which was running the last time the **application was switched off**. The third button `Resume Generation` works with the radio buttons on the right side of the buttons. The radio button specifies from which generation we want to resume our training. The fourth button `Test Best Chromosome` tests the best chromosome that was found in the previous trainings.

**Due to a bug, a button when clicked cannot be deselected. In order to select any other button, please close the GUI application and then select the button**.

## How to perform the exercise?
The student has to edit 3 different functions in `MyAlgorithm.py` file:

- Enter the fitness function in the function `fitness_function`

```python
def fitness_function(self, chromosome):
  # The fitness function
  # Enter the fitness function here
```

- Enter the neural network specifications in the function `define_neural_network`. The input and output layers have to be defined with the given parameters. The student may add additional hidden layers. 

```python
def define_neural_network(self):
  # Define the layers
  inputLayer = Layer("inputLayer", 8, IdentityActivation(), "INFRARED", ["outputLayer"])
  outputLayer = Layer("outputLayer", 2, SigmoidActivation(), "", ["MOTORS"])
  # Define the Neural Network
  neural_network = ArtificialNeuralNetwork([inputLayer, outputLayer], "STATIC")

  return neural_network
```

- Enter the genetic algorithm specifications in the function `define_genetic_algorithm`.  The student mainly has to change the values of the parameters

```python
def define_genetic_algorithm(self):
  # Define the Genetic Algorithm
  neural_network = self.define_neural_network()
  ga = GeneticAlgorithmGazebo(neural_network)

  # Define the genetic algorithm
  log_folder = './log'
  ga.population_size = 
  ga.number_of_generations =    
  ga.mutation_probability = 
  ga.evaluation_time = 
  ga.number_of_elites = 
  ga.fitness_function = self.fitness_function

  genetic_algorithm = GA(ga, log_folder)
```

### Application Programming Interface

- For the specifics of setting the Neural Network, refer to this [API reference](./../libraries/neural_networks/README.md)

- `self.getRange()` to get an array of 8 sensor values

- `self.motors.sendV()` to set the linear speed

- `self.motors.sendW()` to set the angular speed 

## Theory
Coming Soon!

## Hints
Coming Soon!
