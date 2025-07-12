"""
TASK 1: STRING EVOLUTION HILL CLIMBER
-------------------------------------
Implements a simple hill climber algorithm to evolve a target string
"charles darwin was always seasick" from a random initial string.

Key Features:
- Starts with random 33-character string (a-z + space)
- Each generation mutates one random character
- Accepts mutation only if fitness (# correct characters) doesn't decrease
- Efficient fitness calculation (O(1) per generation)
- Prints generational progress
- Stops when target string is matched

Usage:
python hill_climber_string.py
"""

import random

# Target string to evolve towards
target = "charles darwin was always seasick"
n_chars = len(target)

# Generate initial random string
current = ''.join(random.choices("abcdefghijklmnopqrstuvwxyz ", k=n_chars))

# Calculate initial fitness (number of matching characters)
fitness = sum(1 for a, b in zip(current, target) if a == b)

# Initialize generation counter
generation = 0
print(f"Gen {generation}: {current} (Fitness: {fitness})")

# Evolution loop continues until target is reached
while current != target:
    generation += 1  # Increment generation counter
    
    # Select random character position to mutate
    idx = random.randint(0, n_chars-1)
    mutated_list = list(current)
    old_char = mutated_list[idx]
    
    # Generate new random character
    new_char = random.choice("abcdefghijklmnopqrstuvwxyz ")
    
    # Efficient fitness update (O(1) instead of O(n))
    new_fitness = fitness
    if old_char == target[idx]:  # If old char was correct
        new_fitness -= 1         # Fitness decreases
    if new_char == target[idx]:  # If new char is correct
        new_fitness += 1         # Fitness increases
    
    # Hill climbing: accept mutation if fitness doesn't decrease
    if new_fitness >= fitness:
        mutated_list[idx] = new_char
        current = ''.join(mutated_list)
        fitness = new_fitness
    
    # Print progress for each generation
    print(f"Gen {generation}: {current} (Fitness: {fitness})")

# Final success message
print(f"\nTarget reached in {generation} generations!")