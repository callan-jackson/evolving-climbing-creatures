"""
Population module for genetic algorithm.
Manages a collection of creatures and provides selection mechanisms.
"""
import creature
import numpy as np


class Population:
    """Represents a population of creatures for evolutionary computation."""

    def __init__(self, pop_size, gene_count):
        """
        Initialise population with random creatures.

        Args:
            pop_size: Number of creatures in the population
            gene_count: Number of genes per creature genome
        """
        self.creatures = [creature.Creature(gene_count=gene_count)
                          for i in range(pop_size)]

    @staticmethod
    def get_fitness_map(fits):
        """
        Create cumulative fitness map for roulette wheel selection.

        Args:
            fits: List of fitness values for each creature
        Returns:
            Cumulative sum array for probability-based selection
        """
        fitmap = []
        total = 0
        for f in fits:
            total = total + f
            fitmap.append(total)
        return fitmap

    @staticmethod
    def select_parent(fitmap):
        """
        Select parent index using fitness-proportionate selection.

        Uses roulette wheel algorithm where higher fitness creatures
        have proportionally higher chance of being selected.

        Args:
            fitmap: Cumulative fitness map from get_fitness_map()
        Returns:
            Index of selected parent creature
        """
        r = np.random.rand()  # Random value 0-1
        r = r * fitmap[-1]    # Scale to total fitness
        for i in range(len(fitmap)):
            if r <= fitmap[i]:
                return i
