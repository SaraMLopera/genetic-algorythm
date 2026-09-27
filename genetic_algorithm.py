import random
import math
from typing import List, Tuple, Dict


class GeneticAlgorithmNQueens:
    """
    Solves the N-Queens problem using a Genetic Algorithm.
    Chromosome: list of ints of size N where index = column, value = row.
    Fitness: total non-attacking pairs = N*(N-1)/2 - collisions.
    """

    def __init__(
        self,
        n: int = 8,
        population_size: int = 100,
        mutation_rate: float = 0.05,
        elite_size: int = 10,
        max_generations: int = 1000,
        tournament_size: int = 5,
    ):
        self.n = n
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.max_fitness = n * (n - 1) // 2

        self.population: List[List[int]] = []
        self.generation = 0
        self.best_individual: List[int] = []
        self.best_fitness: int = 0
        self.history: List[Dict] = []
        self.solved = False

    # ------------------------------------------------------------------ #
    #  Chromosome helpers
    # ------------------------------------------------------------------ #

    def random_chromosome(self) -> List[int]:
        """A permutation guarantees no row conflicts either (optional – see note)."""
        return random.sample(range(self.n), self.n)

    def fitness(self, chromosome: List[int]) -> int:
        """
        Count collisions (row duplicates + diagonal attacks) and subtract
        from the maximum possible non-attacking pairs.
        """
        collisions = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Same row
                if chromosome[i] == chromosome[j]:
                    collisions += 1
                # Same diagonal
                elif abs(i - j) == abs(chromosome[i] - chromosome[j]):
                    collisions += 1
        return self.max_fitness - collisions

    def count_collisions(self, chromosome: List[int]) -> int:
        return self.max_fitness - self.fitness(chromosome)

    # ------------------------------------------------------------------ #
    #  Population initialisation
    # ------------------------------------------------------------------ #

    def initialize_population(self) -> None:
        self.population = [self.random_chromosome() for _ in range(self.population_size)]
        self.generation = 0
        self.history = []
        self.solved = False
        self._update_best()

    # ------------------------------------------------------------------ #
    #  Selection
    # ------------------------------------------------------------------ #

    def tournament_selection(self) -> List[int]:
        """Pick the best individual from a random tournament subset."""
        contenders = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(contenders, key=self.fitness)

    # ------------------------------------------------------------------ #
    #  Crossover
    # ------------------------------------------------------------------ #

    def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Order Crossover (OX1): preserves relative order from parent2 for
        genes not present in the copied segment, avoiding row duplicates
        when chromosomes are permutations.
        """
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

    def single_point_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        point = random.randint(1, self.n - 1)
        return parent1[:point] + parent2[point:]

    # ------------------------------------------------------------------ #
    #  Mutation
    # ------------------------------------------------------------------ #

    def swap_mutation(self, chromosome: List[int]) -> List[int]:
        """Swap two random positions (preserves permutation property)."""
        child = chromosome[:]
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.n), 2)
            child[i], child[j] = child[j], child[i]
        return child

    def random_reset_mutation(self, chromosome: List[int]) -> List[int]:
        """Assign a random row to a random column (for non-permutation repr.)."""
        child = chromosome[:]
        if random.random() < self.mutation_rate:
            idx = random.randint(0, self.n - 1)
            child[idx] = random.randint(0, self.n - 1)
        return child

    # ------------------------------------------------------------------ #
    #  Evolution step
    # ------------------------------------------------------------------ #

    def _update_best(self) -> None:
        for individual in self.population:
            f = self.fitness(individual)
            if f > self.best_fitness:
                self.best_fitness = f
                self.best_individual = individual[:]
            if f == self.max_fitness:
                self.solved = True

    def evolve_one_generation(self) -> Dict:
        """Run one generation and return stats."""
        fitnesses = [self.fitness(ind) for ind in self.population]
        best_idx = fitnesses.index(max(fitnesses))

        # Elitism – carry the best individuals unchanged
        sorted_pop = [ind for _, ind in sorted(zip(fitnesses, self.population), key=lambda x: -x[0])]
        new_population = sorted_pop[: self.elite_size]

        # Fill the rest
        while len(new_population) < self.population_size:
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            child = self.order_crossover(p1, p2)
            child = self.swap_mutation(child)
            new_population.append(child)

        self.population = new_population
        self.generation += 1
        self._update_best()

        avg_fitness = sum(fitnesses) / len(fitnesses)
        gen_best = max(fitnesses)

        entry = {
            "generation": self.generation,
            "best_fitness": gen_best,
            "avg_fitness": round(avg_fitness, 2),
            "best_overall": self.best_fitness,
            "solved": self.solved,
        }
        self.history.append(entry)
        return entry

    def run(self, callback=None) -> Dict:
        """
        Run the full GA until a solution is found or max_generations reached.
        callback(stats_dict) is called after each generation if provided.
        """
        self.initialize_population()
        for _ in range(self.max_generations):
            stats = self.evolve_one_generation()
            if callback:
                callback(stats)
            if self.solved:
                break

        return {
            "solution": self.best_individual,
            "fitness": self.best_fitness,
            "max_fitness": self.max_fitness,
            "generations": self.generation,
            "solved": self.solved,
            "collisions": self.count_collisions(self.best_individual),
            "history": self.history,
        }

    def get_state(self) -> Dict:
        """Snapshot of current GA state for the frontend."""
        # Top-5 unique individuals by fitness for the population panel
        scored = sorted(
            self.population,
            key=self.fitness,
            reverse=True,
        )
        seen: list[tuple] = []
        top5: list[dict] = []
        for ind in scored:
            key = tuple(ind)
            if key not in seen:
                seen.append(key)
                top5.append({
                    "chromosome": ind[:],
                    "fitness": self.fitness(ind),
                    "collisions": self.count_collisions(ind),
                })
            if len(top5) == 5:
                break

        return {
            "generation": self.generation,
            "best_individual": self.best_individual,
            "best_fitness": self.best_fitness,
            "max_fitness": self.max_fitness,
            "solved": self.solved,
            "collisions": self.count_collisions(self.best_individual) if self.best_individual else 0,
            "history": self.history[-80:],   # keep last 80 for the chart
            "n": self.n,
            "top5": top5,
        }
