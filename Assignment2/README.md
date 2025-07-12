# Evolutionary Robotics - Task Sheet 2

## Completed Tasks
### Task 1: String Hill Climber
* 1a: Implemented in hill_climber_string.py
    * Generates random 33-character string (a-z + space)
    * Mutates one character per generation
    * Accepts mutations only if fitness doesn't decrease
    * Prints current string and fitness each generation

* 1b: Explanation of "good-natured" optimization
    * Smooth fitness landscape with no local optima
    * Independent character optimization
    * Monotonic fitness improvement guaranteed
    * Low-dimensional search space (33 parameters)    
* 1c: Mathematical estimation and empirical validation

### Task 2: Robot Behavior Hill Climber
* Implementation: hill_climber_robot.py
    * Linear controller: v_left = m0*s_left + c0, v_right = m1*s_right + c1 + m2*s_mid + c2
    * 1×1 unit arena with walls
    * Fitness = number of unique 0.01×0.01 grid cells visited
    * Hill climber with Gaussian mutation (σ=0.1)

* Results:
    * Typical fitness improvement: 40-60% over initial random behavior
    * Emerged behaviors: Wall-following, obstacle avoidance
    * Best trajectory plot: trajectory_plot.png
    * Fitness plateaus at ~30% coverage due to local optima

* Interpretation:
    * Effective for basic navigation but struggles with complex spaces
    * Satisfies core requirements but shows algorithm limitations
    * Key limitation: Gets trapped in concave areas

## How to Run
### Dependencies
```console
pip install numpy matplotlib
```
### Task 1: String Hill Climber
```console
python hill_climber_string.py
```
### Output:
* Generational strings printed to console
* Final generation count when target is reached

### Task 2: Robot Behavior Hill Climber
```console
python hill_climber_robot.py
```
### Outputs:
* trajectory_plot.png - Robot path visualization
* best_genome.txt - Best controller parameters
* results.txt - Final fitness and genome
* Console log of fitness improvements

## Results Analysis
### Task 1
* Average generations: 3720 (matches theoretical prediction)
* Fitness increases monotonically
* Confirms problem is "good-natured" with smooth landscape

### Task 2
* Strengths:
    * Successfully evolves wall-avoidance behaviors
    * Clear fitness improvement over time
    * Efficient parameter optimization (6 parameters)
* Limitations:
    * Maximum coverage ≈ 30% of arena
    * Gets stuck in corners and concave spaces
    * Sensitive to initial conditions