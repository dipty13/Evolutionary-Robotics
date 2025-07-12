"""
TASK 2: ROBOT BEHAVIOR HILL CLIMBER
-----------------------------------
Implements a hill climber to evolve robot exploration behavior in a walled environment.

Key Components:
1. Robot Simulation:
   - Differential drive robot with 3 proximity sensors (60° left/forward/right)
   - Wall collision detection and bounce handling
   - Linear controller: v_left = m0*s_left + c0
                       v_right = m1*s_right + c1 + m2*s_mid + c2

2. Evolution:
   - Genome: 6 parameters [m0, c0, m1, c1, m2, c2] in [-1, 1]
   - Mutation: Gaussian noise (σ=0.1) to one random parameter
   - Fitness: Number of unique 0.01×0.01 grid cells visited in 10 seconds
   - Hill climbing: Accepts mutations that don't decrease fitness

3. Visualization:
   - Generates trajectory plot with walls, start/end points
   - Saves best controller parameters and fitness results

Usage:
python hill_climber_robot.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class RobotSimulator:
    def __init__(self, arena_size=(1, 1), grid_size=0.01):
        # Simulation parameters
        self.arena_size = arena_size
        self.grid_size = grid_size
        self.grid_x = int(arena_size[0] / grid_size)  # X grid dimension
        self.grid_y = int(arena_size[1] / grid_size)  # Y grid dimension
        
        # Define wall boundaries and obstacles
        self.walls = [
            # Perimeter walls (bottom, top, left, right)
            {"x": 0, "y": 0, "width": arena_size[0], "height": 0.02},
            {"x": 0, "y": arena_size[1]-0.02, "width": arena_size[0], "height": 0.02},
            {"x": 0, "y": 0, "width": 0.02, "height": arena_size[1]},
            {"x": arena_size[0]-0.02, "y": 0, "width": 0.02, "height": arena_size[1]},
            
            # Internal obstacles
            {"x": 0.3, "y": 0.3, "width": 0.02, "height": 0.4},  # Vertical wall
            {"x": 0.5, "y": 0.5, "width": 0.4, "height": 0.02},  # Horizontal wall
        ]
    
    def get_sensor_readings(self, x, y, theta):
        """Calculate proximity sensor readings (left, middle, right)"""
        sensor_angles = [np.pi/3, 0, -np.pi/3]  # 60° left, forward, 60° right
        readings = [0, 0, 0]
        
        for i, angle in enumerate(sensor_angles):
            sensor_dir = theta + angle  # Absolute sensor direction
            min_distance = float('inf')  # Initialize with large value
            
            # Calculate distance to all walls
            for wall in self.walls:
                # Horizontal wall detection
                if wall["height"] < wall["width"]:
                    y_wall = wall["y"]
                    # Calculate intersection point
                    x_intersect = x + (y_wall - y) / np.tan(sensor_dir)
                    if (wall["x"] <= x_intersect <= wall["x"] + wall["width"] and
                            min(y, y_wall) <= y_wall <= max(y, y_wall)):
                        distance = np.sqrt((x_intersect - x)**2 + (y_wall - y)**2)
                        min_distance = min(min_distance, distance)
                
                # Vertical wall detection
                else:
                    x_wall = wall["x"]
                    y_intersect = y + (x_wall - x) * np.tan(sensor_dir)
                    if (wall["y"] <= y_intersect <= wall["y"] + wall["height"] and
                            min(x, x_wall) <= x_wall <= max(x, x_wall)):
                        distance = np.sqrt((x_wall - x)**2 + (y_intersect - y)**2)
                        min_distance = min(min_distance, distance)
            
            # Normalize sensor reading (1 = close, 0 = far)
            readings[i] = max(0, 1 - min(min_distance, 1.0))
        
        return readings
    
    def is_collision(self, x, y):
        """Check if robot collides with any wall"""
        for wall in self.walls:
            if (wall["x"] <= x <= wall["x"] + wall["width"] and
                    wall["y"] <= y <= wall["y"] + wall["height"]):
                return True
        return False
    
    def simulate_robot(self, genome, duration=10, dt=0.1, init_pos=(0.1, 0.1, 0)):
        """
        Simulate robot with given controller genome
        genome: [m0, c0, m1, c1, m2, c2] controller parameters
        Returns: 
            fitness: number of unique grid cells visited
            trajectory: list of (x,y) positions
        """
        # Initial position and orientation
        x, y, theta = init_pos
        visited = np.zeros((self.grid_x, self.grid_y), dtype=bool)  # Coverage grid
        trajectory = []  # Path history
        wheel_base = 0.1  # Distance between wheels (10cm)
        
        # Unpack controller parameters
        m0, c0, m1, c1, m2, c2 = genome
        
        # Simulation loop
        for t in np.arange(0, duration, dt):
            # Get sensor readings
            s_left, s_mid, s_right = self.get_sensor_readings(x, y, theta)
            
            # Calculate wheel speeds using linear controller
            v_left = m0 * s_left + c0
            v_right = m1 * s_right + c1 + m2 * s_mid + c2
            
            # Calculate overall motion
            v = (v_left + v_right) / 2          # Linear velocity
            omega = (v_right - v_left) / wheel_base  # Angular velocity
            
            # Update orientation
            theta += omega * dt
            # Calculate new position
            new_x = x + v * np.cos(theta) * dt
            new_y = y + v * np.sin(theta) * dt
            
            # Ensure position stays within arena bounds
            new_x = max(0.01, min(0.99, new_x))
            new_y = max(0.01, min(0.99, new_y))
            
            # Collision handling
            if not self.is_collision(new_x, new_y):
                x, y = new_x, new_y  # Accept new position
            else:
                # Random bounce on collision
                theta += np.random.uniform(np.pi/4, 3*np.pi/4)
                # Ensure valid position after bounce
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
            
            # Convert position to grid coordinates
            grid_x = int(x / self.grid_size)
            grid_y = int(y / self.grid_size)
            # Clamp grid indices to valid range
            grid_x = max(0, min(grid_x, self.grid_x - 1))
            grid_y = max(0, min(grid_y, self.grid_y - 1))
            
            # Mark grid cell as visited
            visited[grid_x, grid_y] = True
            trajectory.append((x, y))  # Record position
        
        return np.sum(visited), trajectory  # Fitness = visited cell count

class HillClimber:
    def __init__(self, param_range=(-1, 1)):
        self.param_range = param_range  # Genome parameter bounds
        self.simulator = RobotSimulator()  # Simulation environment
    
    def random_genome(self):
        """Generate random controller parameters"""
        return np.random.uniform(*self.param_range, size=6)
    
    def mutate(self, genome, mutation_std=0.1):
        """Mutate one randomly selected parameter"""
        mutated = genome.copy()
        idx = np.random.randint(len(genome))  # Choose random parameter
        # Apply Gaussian mutation
        mutated[idx] += np.random.normal(0, mutation_std)
        # Ensure parameter stays within bounds
        mutated[idx] = np.clip(mutated[idx], *self.param_range)
        return mutated
    
    def evolve(self, generations=100):
        """Main evolution loop"""
        # Initialize random controller
        current_genome = self.random_genome()
        current_fitness, trajectory = self.simulator.simulate_robot(current_genome)
        best_fitness = current_fitness
        best_genome = current_genome.copy()
        best_trajectory = trajectory
        
        print(f"Initial fitness: {current_fitness}")
        
        # Evolution loop
        for gen in range(generations):
            # Create mutated candidate
            candidate = self.mutate(current_genome)
            # Evaluate candidate
            candidate_fitness, candidate_trajectory = self.simulator.simulate_robot(candidate)
            
            # Hill climbing: accept candidate if fitness doesn't decrease
            if candidate_fitness >= current_fitness:
                current_genome = candidate
                current_fitness = candidate_fitness
                
                # Update best solution found so far
                if candidate_fitness > best_fitness:
                    best_fitness = candidate_fitness
                    best_genome = candidate.copy()
                    best_trajectory = candidate_trajectory
                    print(f"Generation {gen}: New best fitness = {best_fitness}")
        
        return best_genome, best_fitness, best_trajectory

def plot_trajectory(trajectory, walls, filename="trajectory_plot.png"):
    """Visualize robot path with walls"""
    plt.figure(figsize=(10, 10))
    x, y = zip(*trajectory)
    plt.plot(x, y, 'b-', alpha=0.5)  # Path
    plt.scatter(x[0], y[0], c='green', marker='o', s=100, label='Start')  # Start
    plt.scatter(x[-1], y[-1], c='red', marker='x', s=100, label='End')  # End
    
    # Draw walls
    for wall in walls:
        plt.gca().add_patch(Rectangle(
            (wall["x"], wall["y"]), wall["width"], wall["height"],
            facecolor='gray', edgecolor='black'
        ))
    
    # Plot configuration
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Robot Exploration Trajectory with Walls")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Initialize and run evolution
    hc = HillClimber()
    best_genome, best_fitness, trajectory = hc.evolve(generations=200)
    
    # Print results
    print(f"\nBest fitness achieved: {best_fitness}")
    print(f"Best genome: {best_genome}")
    
    # Visualize best trajectory
    simulator = RobotSimulator()
    plot_trajectory(trajectory, simulator.walls)
    
    # Save results
    np.savetxt("best_genome.txt", best_genome)
    with open("results.txt", "w") as f:
        f.write(f"Best Fitness: {best_fitness}\n")
        f.write(f"Best Genome: {best_genome}\n")