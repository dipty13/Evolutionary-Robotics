import numpy as np
import matplotlib.pyplot as plt

"""
Evolutionary ANN for Classification

This implementation evolves a simple artificial neural network (ANN) to solve a binary classification problem.
"""

def load_data(filename='data.txt'):
    """Loading dataset from file"""
    data = np.loadtxt(filename, usecols=(1,2,3))  # Skip index column
    return data[:,1:], data[:,0]  # Features, classes

def activation(x):
    """Tanh-like activation function"""
    return 2 / (1 + np.exp(-2*x)) - 1

class SimpleANN:
    """Single-layer perceptron with 2 inputs, 1 output, and bias"""
    def __init__(self, w0=None, w1=None, w2=None):
        # Initialize weights randomly if not provided
        self.w0 = np.random.normal(0,1) if w0 is None else w0  # bias
        self.w1 = np.random.normal(0,1) if w1 is None else w1  # x weight
        self.w2 = np.random.normal(0,1) if w2 is None else w2  # y weight
        self.fitness = 0
    
    def predict(self, x, y):
        """Making prediction for input features"""
        return activation(self.w0 + self.w1*x + self.w2*y)
    
    def evaluate(self, features, classes):
        """Calculating classification accuracy"""
        correct = 0
        for (x,y), cls in zip(features, classes):
            output = self.predict(x,y)
            if (output > 0 and cls == 1) or (output < 0 and cls == 0):
                correct += 1
        self.fitness = correct / len(classes)
        return self.fitness
    
    def mutate(self, rate=0.1):
        """Mutating weights with Gaussian noise"""
        if np.random.random() < rate:
            self.w0 += np.random.normal(0,0.1)
        if np.random.random() < rate:
            self.w1 += np.random.normal(0,0.1)
        if np.random.random() < rate:
            self.w2 += np.random.normal(0,0.1)

def evolve_ann(features, classes, pop_size=50, generations=100):
    """Evolve ANN population"""
    population = [SimpleANN() for _ in range(pop_size)]
    best_fitness = []
    avg_fitness = []
    
    for gen in range(generations):
        # Evaluate population
        for ann in population:
            ann.evaluate(features, classes)
        
        # Record stats
        fitnesses = [ann.fitness for ann in population]
        best_fitness.append(max(fitnesses))
        avg_fitness.append(np.mean(fitnesses))
        
        # Create new generation
        population.sort(key=lambda x: -x.fitness)
        new_pop = population[:1]  # Keep best
        
        while len(new_pop) < pop_size:
            # Tournament selection
            parents = np.random.choice(population[:pop_size//2], 2, replace=False)
            # Create child (simplified crossover)
            child = SimpleANN(
                parents[0].w0 if np.random.random() < 0.5 else parents[1].w0,
                parents[0].w1 if np.random.random() < 0.5 else parents[1].w1,
                parents[0].w2 if np.random.random() < 0.5 else parents[1].w2
            )
            child.mutate()
            new_pop.append(child)
        
        population = new_pop
    
    # Plot results
    plt.figure(figsize=(10,6))
    plt.plot(best_fitness, label='Best Fitness')
    plt.plot(avg_fitness, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.title('ANN Training Progress')
    plt.legend()
    plt.grid()
    plt.savefig('ann_fitness.png')
    plt.show()
    
    return max(population, key=lambda x: x.fitness)

def plot_decision_boundary(ann, features, classes):
    """Visualizing classification results"""
    plt.figure(figsize=(10,8))
    
    # Ploting data points
    class0 = features[classes == 0]
    class1 = features[classes == 1]
    plt.scatter(class0[:,0], class0[:,1], label='Class 0')
    plt.scatter(class1[:,0], class1[:,1], label='Class 1')
    
    # Ploting decision boundary (w0 + w1*x + w2*y = 0)
    x_vals = np.array([features[:,0].min(), features[:,0].max()])
    y_vals = (-ann.w0 - ann.w1*x_vals) / ann.w2
    plt.plot(x_vals, y_vals, 'k-', label='Decision Boundary')
    
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.title('ANN Classification Results')
    plt.legend()
    plt.grid()
    plt.savefig('decision_boundary.png')
    plt.show()

if __name__ == "__main__":
    features, classes = load_data()
    best_ann = evolve_ann(features, classes)
    
    print(f"Best ANN weights:")
    print(f"w0 (bias): {best_ann.w0:.4f}")
    print(f"w1 (x weight): {best_ann.w1:.4f}")
    print(f"w2 (y weight): {best_ann.w2:.4f}")
    print(f"Accuracy: {best_ann.fitness*100:.2f}%")
    
    plot_decision_boundary(best_ann, features, classes)