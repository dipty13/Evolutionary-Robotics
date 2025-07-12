import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

"""
For the more complex dataset, I added:"
* Hidden Layer: A 6-neuron hidden layer with tanh activation enables non-linear decision boundaries.
* Enhanced Operators: Xavier initialization and adaptive mutation improve evolution.
"""

def load_data(filename='data2.txt'):
    """Loading enhanced dataset"""
    data = np.loadtxt(filename, usecols=(1,2,3))
    return data[:,1:], data[:,0]

def sigmoid(x):
    """Sigmoid activation for output layer"""
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    """Tanh-like activation for hidden layer"""
    return 2 / (1 + np.exp(-2*x)) - 1

class EnhancedANN:
    """ANN with one hidden layer and bias"""
    def __init__(self, hidden_size=4):
        # Xavier initialization
        self.w1 = np.random.randn(3, hidden_size) * np.sqrt(2./3)  # input to hidden (with bias)
        self.w2 = np.random.randn(hidden_size+1, 1) * np.sqrt(2./(hidden_size+1))  # hidden to output (with bias)
        self.fitness = 0
    
    def predict(self, x):
        """Forward pass through network"""
        # Add bias and compute hidden layer
        x_bias = np.append(x, 1)
        hidden = tanh_activation(x_bias @ self.w1)
        
        # Add bias and compute output
        hidden_bias = np.append(hidden, 1)
        return sigmoid(hidden_bias @ self.w2)[0]
    
    def evaluate(self, features, classes):
        """Calculating classification accuracy"""
        correct = 0
        for x, cls in zip(features, classes):
            output = self.predict(x)
            if (output > 0.5 and cls == 1) or (output <= 0.5 and cls == 0):
                correct += 1
        self.fitness = correct / len(classes)
        return self.fitness
    
    def mutate(self, rate=0.1, scale=0.2):
        """Mutating weights with adaptive scaling"""
        self.w1 += (np.random.random(self.w1.shape) < rate) * np.random.normal(0, scale, self.w1.shape)
        self.w2 += (np.random.random(self.w2.shape) < rate) * np.random.normal(0, scale, self.w2.shape)

def evolve_enhanced_ann(features, classes, pop_size=80, generations=150, hidden_size=6):
    """Evolving population of enhanced ANNs"""
    population = [EnhancedANN(hidden_size) for _ in range(pop_size)]
    best_fitness = []
    avg_fitness = []
    
    for gen in range(generations):
        # Evaluate population
        for ann in population:
            ann.evaluate(features, classes)
        
        # Record stats
        population.sort(key=lambda x: -x.fitness)
        best_fitness.append(population[0].fitness)
        avg_fitness.append(np.mean([ann.fitness for ann in population]))
        
        # Create new generation with elitism
        new_pop = population[:2]  # Keep top 2
        
        while len(new_pop) < pop_size:
            # Tournament selection from top 50%
            parents = np.random.choice(population[:pop_size//2], 2, replace=False)
            # Creating child (simplified crossover)
            child = EnhancedANN(hidden_size)
            # Uniform crossover on flattened weights
            w1_flat = np.concatenate([parents[0].w1.flatten(), parents[0].w2.flatten()])
            w2_flat = np.concatenate([parents[1].w1.flatten(), parents[1].w2.flatten()])
            mask = np.random.randint(0, 2, w1_flat.shape).astype(bool)
            child_weights = np.where(mask, w1_flat, w2_flat)
            # Reshape and assign weights
            w1_size = parents[0].w1.size
            child.w1 = child_weights[:w1_size].reshape(parents[0].w1.shape)
            child.w2 = child_weights[w1_size:].reshape(parents[0].w2.shape)
            child.mutate()
            new_pop.append(child)
        
        population = new_pop
    
    # Ploting training progress
    plt.figure(figsize=(10,6))
    plt.plot(best_fitness, label='Best Fitness')
    plt.plot(avg_fitness, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.title('Enhanced ANN Training')
    plt.legend()
    plt.grid()
    plt.savefig('enhanced_ann_fitness.png')
    plt.show()
    
    return population[0]  # Return best ANN

def plot_decision_boundary(ann, features, classes, resolution=0.02):
    """Ploting non-linear decision boundary"""
    # Set up grid
    x_min, x_max = features[:,0].min()-0.1, features[:,0].max()+0.1
    y_min, y_max = features[:,1].min()-0.1, features[:,1].max()+0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # Predicting for each grid point
    Z = np.array([ann.predict(np.array([x,y])) for x,y in zip(xx.ravel(), yy.ravel())])
    Z = (Z > 0.5).reshape(xx.shape)
    
    # Creating plot
    plt.figure(figsize=(10,8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(['#FFAAAA', '#AAFFAA']))
    plt.scatter(features[:,0], features[:,1], c=classes, edgecolor='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f'Enhanced ANN Decision Boundary (Hidden Units: {ann.w1.shape[1]})')
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.savefig('enhanced_decision_boundary.png')
    plt.show()

if __name__ == "__main__":
    features, classes = load_data()
    best_ann = evolve_enhanced_ann(features, classes)
    
    print(f"Best Enhanced ANN achieved {best_ann.fitness*100:.2f}% accuracy")
    print("Sample weights from first hidden neuron:")
    print(f"Input to hidden: {best_ann.w1[:,0]}")
    print(f"Hidden to output: {best_ann.w2[0,:]}")
    
    plot_decision_boundary(best_ann, features, classes)