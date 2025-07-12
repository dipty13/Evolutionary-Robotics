import numpy as np
import matplotlib.pyplot as plt
from math import exp, sqrt, cos, pi

"""
Evolutionary Algorithm for Ackley Function Optimization
This implementation solves the 3D Ackley function minimization problem using an evolutionary algorithm.
"""

# Ackley function implementation
def ackley(x, y, z):
    """
    Calculating the Ackley function value for given x, y, z coordinates.
    This function has many local minima, which makes it challenging for optimization.
    """
    term1 = -20 * exp(-0.2 * sqrt(1/3 * (x**2 + y**2 + z**2)))
    term2 = -exp(1/3 * (cos(2*pi*x) + cos(2*pi*y) + cos(2*pi*z)))
    return term1 + term2 + 20 + exp(1)

# Fitness function for converting minimization to maximization problem
def fitness(x, y, z):
    """
    Converting Ackley function output to fitness value.
    Added 1 to avoid division by zero and convert minimization to maximization.
    """
    return 1 / (ackley(x, y, z) + 1)

class Individual:
    """
    Representing an individual in the population with genes (x, y, z coordinates)
    and a fitness value.
    """
    def __init__(self, x=None, y=None, z=None):
        # Initialize with random values if none provided
        if x is None:
            self.x = np.random.uniform(-32.768, 32.768)
            self.y = np.random.uniform(-32.768, 32.768)
            self.z = np.random.uniform(-32.768, 32.768)
        else:
            self.x = x
            self.y = y
            self.z = z
        self.fitness = fitness(self.x, self.y, self.z)
    
    def mutate(self, mutation_rate):
        """
        Applying Gaussian mutation to each gene with given probability.
        Mutation helps explore the search space.
        """
        if np.random.random() < mutation_rate:
            self.x += np.random.normal(0, 1)
            # Ensure values stay within bounds
            self.x = np.clip(self.x, -32.768, 32.768)
        if np.random.random() < mutation_rate:
            self.y += np.random.normal(0, 1)
            self.y = np.clip(self.y, -32.768, 32.768)
        if np.random.random() < mutation_rate:
            self.z += np.random.normal(0, 1)
            self.z = np.clip(self.z, -32.768, 32.768)
        self.fitness = fitness(self.x, self.y, self.z)

def crossover(parent1, parent2):
    """
    Performing uniform crossover between two parents to create a child.
    Each gene has 50% chance to come from either parent.
    """
    child = Individual()
    if np.random.random() < 0.5:
        child.x = parent1.x
    else:
        child.x = parent2.x
    
    if np.random.random() < 0.5:
        child.y = parent1.y
    else:
        child.y = parent2.y
    
    if np.random.random() < 0.5:
        child.z = parent1.z
    else:
        child.z = parent2.z
    
    return child

def tournament_selection(population, tournament_size=3):
    """
    Selecting an individual using tournament selection.
    Randomly picking tournament_size individuals and returning the fittest one.
    """
    participants = np.random.choice(population, tournament_size, replace=False)
    return max(participants, key=lambda ind: ind.fitness)

def evolutionary_algorithm(pop_size=50, generations=100, mutation_rate=0.1, elitism=1):
    """
    Main evolutionary algorithm implementation.
    """
    # Initializing population
    population = [Individual() for _ in range(pop_size)]
    best_fitness = []
    avg_fitness = []
    
    for gen in range(generations):
        # Evaluating current population
        fitnesses = [ind.fitness for ind in population]
        best_fitness.append(max(fitnesses))
        avg_fitness.append(np.mean(fitnesses))
        
        # Creating next generation
        new_population = []
        
        # Elitism: keep the best individual(s)
        elites = sorted(population, key=lambda ind: ind.fitness, reverse=True)[:elitism]
        new_population.extend(elites)
        
        # Filling the rest of the population with offspring
        while len(new_population) < pop_size:
            # Selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child.mutate(mutation_rate)
            
            new_population.append(child)
        
        population = new_population
    
    # Ploting results
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness, label='Best Fitness')
    plt.plot(avg_fitness, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolutionary Algorithm Performance')
    plt.legend()
    plt.grid()
    plt.savefig('ackley_fitness.png')
    plt.show()
    
    # Returning best individual
    best_individual = max(population, key=lambda ind: ind.fitness)
    return best_individual

# Runing the algorithm with default parameters
best_ind = evolutionary_algorithm()
print(f"Best solution found: x={best_ind.x}, y={best_ind.y}, z={best_ind.z}")
print(f"Fitness: {best_ind.fitness}, Ackley value: {ackley(best_ind.x, best_ind.y, best_ind.z)}")