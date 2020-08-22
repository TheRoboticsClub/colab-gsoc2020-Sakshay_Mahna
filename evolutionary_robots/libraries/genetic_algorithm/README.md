# Genetic Algorithm API

## How to generate a Genetic Algorithm object

**Import the Genetic Algorithm** along with Numpy(optional). The `genetic_algorithm.ga` consists of the Genetic Algorithm object as GeneticAlgorithm class

```python
from genetic_algorithm.ga import GeneticAlgorithm
import numpy as np
```

**Design the fitness function** The fitness function is supposed to calculate the fitness of a given chromosome. For it, the function is allowed to take a single argument which is a numpy list object. For our example, suppose the fitness value is the sum of the alleles of the chromosome

```python
def fitness_function(chromosome):
	return np.sum(chromosome)
```

**Initialize the GeneticAlgorithm object** For our specefic example, the size of population is assumed to be 100, the number of generations are 50, the chromosome length is 3.

```python
ga = GeneticAlgorithm()
ga.population_size = 100
ga.number_of_generations = 50
ga.chromosome_length = 3
```

**Simulate the algorithm and plot the results**

```python
best_chromosome = ga.run()
print(best_chromosome)

ga.plot()
```

The [examples directory](./../examples) contains [ga_sum.py](./../examples/ga_sum.py) example that maximizes the sum of the allele values using Genetic Algorithm.

## Genetic Algorithm

`ga.py` contains the Genetic Algorithm class which allows us to create the algorithm object. The following parameters of the Genetic Algorithm can be adjusted:

- Size of Population
- Number of Generations
- Probability of Mutation
- Length of Chromosome
- Number of Elites
- Fitness Function

The fitness function is a crucial parameters, without which the algorithm would not work. The Genetic Algorithm can be used as follows

**Initialization** The parameters of the algorithm can be passed during initialization or by changing the attributes

```python
# Initializing Genetic Algorithm object
ga = GeneticAlgorithm(population_size, number_of_generations, mutation_probability, chromosome_length, number_of_elites)

# Initializing and then changing the attributes
ga = GeneticAlgorithm()
ga.population_size = population_size
ga.number_of_generations = number_of_generations
ga.mutation_probability = mutation_probability
ga.chromosome_length = chromosome_length
ga.number_of_elites = number_of_elites
```

**population_size** specifies the size of the population for each generation. The default value of the population size is 100. **The population size should be an even number**.

**number_of_generations** specifies the number of generations for which the genetic algorithm would run. By default the algorithm runs for 10 generations.

**mutation_probability** specifies the probability of mutation. This attribute should lie between 0 and 1. The default value of this attribute is 0.01

**chromosome_length** specifies the length of the chromosome of each individual. By default, the length of chromosome is taken to be 5.

**number_of_elites** specifies the number of elites in the algorithm. Elites are not crossovered and directly sent to the next generation. By default, the algorithm runs with 0 elites. **The parity of number of elites and size of population should be same, otherwise the result would be an error**.

**replay_number** specifies the interval through which generations to save. This saves all the chromosomes of the generation which occur at intervals of replay number. By default, the value of this attribute is 25. An example as shown:

```python
ga.replay_number = 30
```

**Specifiying Fitness Function** The fitness function is set as an attribute for the algorithm. The fitness function variable should be a function object, that takes in a single parameter, which is a numpy list object and returns a single comparable(float or integer) value. In essence, the user-defined fitness function should be able to calculate and return the fitness value of a single chromosome.

```python
ga.fitness_function = fitness_function
```

**Running and Plotting** 

```python
ga.run()
ga.plot_fitness()
```

`ga.run()` function runs a simulation of the genetic algorithm for the specifed number of generations. It also prints some statistics regarding the minimum, maximum and the average fitness values. The function returns the chromosomes with the best fitness value for the whole simulation. Along with it, these statistics are saved in `stats.txt`. The best chromosomes of each generation are stored in `best_chromosomes.txt`, all the chromosomes of the current generation and generations according to the replay number attribute are saved as `generations<generation_number>.txt`. These represent the populalation when the algorithm has reached certain percentage of total generations to run.

These files are saved in a directory named as `log`.

`ga.plot_fitness()` generates a matplotlib plot of the fitness value as a function of the generation.
