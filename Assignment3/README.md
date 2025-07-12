# Evolutionary Algorithms Implementation for Task Sheet 3
## Overview
This repository contains solutions for both Task 1 (Ackley function optimization) and Task 2 (ANN classification) from the Evolutionary Robotics assignment, including all optional parts. The implementation uses Python with NumPy and Matplotlib.

### Task 1: Ackley Function Optimization
* Objective: Minimize the 3D Ackley function using evolutionary algorithms.
* Implementation:
    * Genetic representation: 3D coordinates (x,y,z)
    * Population size: 50 individuals
    * Selection: Tournament selection (size 3)
    * Replacement: Generational with elitism (keep best individual)
    * Mutation: Gaussian mutation with rate 0.1
    * Crossover: Uniform crossover
    * Fitness: 1/(Ackley(x,y,z) + 1)
#### Files:
* ackley_optimization.py: Main implementation
* ackley_fitness.png: Fitness progression plot   

### Task 2: ANN Classification
* Objective: Evolve ANNs for binary classification.
* Basic Implementation (for data.txt):
    * Simple perceptron (2 inputs, 1 output, bias)
    * Activation: tanh-like function
    * Fitness: Classification accuracy
* Enhanced Implementation (for data2.txt - optional):    
    * 2-6-1 ANN with hidden layer
    * Both sigmoid and tanh activations
    * Bias neurons for both layers

#### Files:
* ann_classification.py: Basic ANN implementation
* enhanced_ann.py: Enhanced ANN with hidden layer
* decision_boundary.png: Basic ANN decision plot
* enhanced_decision_boundary.png: Enhanced ANN decision plot
* ann_fitness.png: Basic ANN fitness plot
* enhanced_ann_fitness.png: Enhanced ANN fitness plot
### Requirements
* Python 3.6+
* NumPy
* Matplotlib

### How to Run
nstall requirements:
```console
pip install numpy matplotlib
````
In the terminal swith to the correct folder for Task1:
```console
cd Task1
````
Run the code:
```console
python ackley_optimization.py
````

In the terminal swith to the correct folder for Task2:
```console
cd ..
cd Task2
````
#### Run the code
For Task 2 basic: 
```console
python ann_classification.py
````
For Task 2 enhanced: 
```console
python enhanced_ann.py
````
## Results
Both tasks show successful convergence:

* Ackley optimization finds good solutions within 100 generations.
* Basic ANN achieves >90% accuracy on simple dataset.
* Enhanced ANN achieves >85% accuracy on complex dataset.