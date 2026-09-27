# Genetic Algorithm for the N-Queens Problem

A web-based implementation of the N-Queens problem solved using a genetic algorithm (GA). The project combines evolutionary optimization with an interactive user interface that allows configuring parameters, observing the population evolution, and visualizing the best candidate solution in real time.

## Authors

- Sara Monsalve Lopera
- Andrés Felipe Mesa Colorado

## Overview

The N-Queens problem consists of placing N queens on an N x N chessboard so that no two queens attack each other. In other words, no two queens may share the same row, column, or diagonal.

This project addresses the challenge through a genetic algorithm, where each candidate solution is represented as a chromosome that encodes the position of each queen in the board. The algorithm iteratively evolves a population of candidate solutions by applying selection, crossover, and mutation operations until it finds a valid board arrangement or reaches the maximum number of generations.

## Features

- Solves the N-Queens problem using genetic optimization
- Allows dynamic configuration of algorithm parameters
- Provides a visual board representation of the current best solution
- Shows generation statistics such as fitness and collisions
- Implements a step-by-step execution mode and real-time streaming mode
- Includes a web interface built with Flask and HTML/JavaScript

## Tech Stack

- Python 3
- Flask
- JavaScript
- HTML/CSS
- Chart.js for visual analytics

## Project Architecture

The project is organized into a lightweight three-layer architecture:

### 1. Evolution engine

The core logic is implemented in `genetic_algorithm.py`. This file contains the `GeneticAlgorithmNQueens` class, which is responsible for:

- chromosome generation
- fitness evaluation
- tournament selection
- crossover
- mutation
- population evolution
- solution validation

### 2. Web API

The file `app.py` defines the Flask application and exposes REST endpoints to:

- initialize a new search
- advance one generation
- stream generations continuously
- stop the execution
- retrieve the current GA state

### 3. Frontend

The interface in `templates/index.html` provides the UI for:

- configuring parameters
- starting, stopping, and stepping the algorithm
- rendering the chessboard
- displaying metrics and historical evolution

## Project Structure

```text
Algoritmo Genetico/
├── app.py
├── genetic_algorithm.py
├── requirements.txt
├── README.md
└── templates/
    └── index.html
```

## Important Code Blocks

### Genetic representation

Each chromosome is a permutation of rows, where each index represents a column and each value represents the row of the queen:

```python
class GeneticAlgorithmNQueens:
    def random_chromosome(self) -> List[int]:
        return random.sample(range(self.n), self.n)
```

Example:

```python
[2, 5, 7, 1, 3, 6, 4, 0]
```

### Fitness function

The algorithm evaluates a candidate solution based on the number of pairwise conflicts between queens:

```python
def fitness(self, chromosome: List[int]) -> int:
    collisions = 0
    for i in range(self.n):
        for j in range(i + 1, self.n):
            if chromosome[i] == chromosome[j]:
                collisions += 1
            elif abs(i - j) == abs(chromosome[i] - chromosome[j]):
                collisions += 1
    return self.max_fitness - collisions
```

A higher fitness value indicates a better solution. The ideal fitness corresponds to a conflict-free arrangement.

### Tournament selection

This method selects parents based on the strongest candidates within a random subset:

```python
def tournament_selection(self) -> List[int]:
    contenders = random.sample(self.population, min(self.tournament_size, len(self.population)))
    return max(contenders, key=self.fitness)
```

### Crossover

The algorithm uses order crossover to preserve valid permutations without duplicating rows:

```python
def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
    size = self.n
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end + 1] = parent1[start:end + 1]

    remaining = [gene for gene in parent2 if gene not in child]
    idx = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = remaining[idx]
            idx += 1
    return child
```

### Mutation

Mutation introduces diversity and prevents the population from prematurely converging to a poor local optimum:

```python
def swap_mutation(self, chromosome: List[int]) -> List[int]:
    child = chromosome[:]
    if random.random() < self.mutation_rate:
        i, j = random.sample(range(self.n), 2)
        child[i], child[j] = child[j], child[i]
    return child
```

### Evolution cycle

The main evolution loop updates the population generation by generation:

```python
def evolve_one_generation(self) -> Dict:
    fitnesses = [self.fitness(ind) for ind in self.population]
    sorted_pop = [ind for _, ind in sorted(zip(fitnesses, self.population), key=lambda x: -x[0])]
    new_population = sorted_pop[: self.elite_size]

    while len(new_population) < self.population_size:
        p1 = self.tournament_selection()
        p2 = self.tournament_selection()
        child = self.order_crossover(p1, p2)
        child = self.swap_mutation(child)
        new_population.append(child)

    self.population = new_population
    self.generation += 1
    self._update_best()
```

## How the Algorithm Works

1. A random initial population is generated.
2. Each chromosome is evaluated using the fitness function.
3. The best individuals are selected for reproduction.
4. Offspring are produced through crossover.
5. Mutation is applied to maintain diversity.
6. The process is repeated until a valid solution is found or the generation limit is reached.

## Installation and Execution

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone the project

```bash
git clone <repository-url>
cd "Algoritmo Genetico"
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

On Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python app.py
```

Then open the following URL in your browser:

```text
http://127.0.0.1:5000
```

## Configuration Parameters

The genetic algorithm can be adjusted using the following parameters:

- `n`: board size and number of queens
- `population_size`: population size per generation
- `mutation_rate`: probability of mutation
- `elite_size`: number of best individuals preserved each generation
- `max_generations`: maximum number of iterations
- `tournament_size`: number of individuals used in tournament selection

Example:

```python
algorithm = GeneticAlgorithmNQueens(
    n=8,
    population_size=120,
    mutation_rate=0.08,
    elite_size=12,
    max_generations=2000,
    tournament_size=5,
)
```

## Use Cases

This project is useful for:

- learning how genetic algorithms work
- exploring optimization techniques applied to combinatorial problems
- studying convergence behavior in evolutionary computation
- visualizing the impact of crossover and mutation on solution quality

## Future Improvements

- Add alternative crossover strategies
- Implement additional selection methods
- Improve the visualization of the board and convergence metrics
- Save results to CSV or JSON files
- Add support for larger board configurations and more advanced optimization strategies

## Conclusion

This project demonstrates an effective and intuitive application of genetic algorithms to the N-Queens problem. It combines theoretical optimization concepts with a practical visual interface, making it suitable for academic, educational, and experimentation purposes.

## Credits

Developed by:

- Sara Monsalve Lopera
- Andrés Felipe Mesa Colorado
