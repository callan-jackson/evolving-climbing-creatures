"""
Master Evolution Script for Mountain Climbing Creatures

This script runs the genetic algorithm to evolve creatures capable of
climbing the Gaussian pyramid mountain environment.

Usage:
    python master_evolution.py
"""

import population
import genome
import creature
import numpy as np
import os
from mountain_simulation import MountainThreadedSim


def run_evolution(
    pop_size=20,
    gene_count=5,
    generations=50,
    sim_steps=2400,
    pool_size=4,
    mutation_rate=0.1,
    mutation_amount=0.25,
    shrink_rate=0.25,
    grow_rate=0.1,
    output_dir="evolution_output"
):
    """
    Run the genetic algorithm to evolve mountain climbing creatures.

    Args:
        pop_size: Number of creatures in population
        gene_count: Number of genes per creature (affects complexity)
        generations: Number of evolutionary generations
        sim_steps: Physics simulation steps per evaluation (240 = 1 second)
        pool_size: Number of parallel simulation processes
        mutation_rate: Probability of mutating each gene value
        mutation_amount: Maximum mutation magnitude
        shrink_rate: Probability of removing a gene
        grow_rate: Probability of adding a gene
        output_dir: Directory to save elite DNA files
    """

    # Create output directory for elite DNA files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize population
    print(f"Initializing population: {pop_size} creatures, {gene_count} genes each")
    pop = population.Population(pop_size=pop_size, gene_count=gene_count)

    # Initialize threaded simulation
    print(f"Starting threaded simulation with {pool_size} workers")
    sim = MountainThreadedSim(pool_size=pool_size)

    # Track best fitness across all generations
    all_time_best_fitness = 0
    all_time_best_dna = None

    print(f"\nStarting evolution for {generations} generations...")
    print("-" * 60)

    for gen in range(generations):
        # 1. Evaluate population fitness
        sim.eval_population(pop, sim_steps)

        # 2. Collect fitness values
        fits = [cr.fitness for cr in pop.creatures]
        links = [len(cr.get_expanded_links()) for cr in pop.creatures]

        # 3. Statistics
        max_fit = np.max(fits)
        mean_fit = np.mean(fits)
        best_idx = np.argmax(fits)

        print(f"Gen {gen:3d}: Max: {max_fit:.3f}, Mean: {mean_fit:.3f}, "
              f"Links: {np.mean(links):.1f} avg / {np.max(links)} max")

        # 4. Save elite DNA
        elite_dna = pop.creatures[best_idx].dna
        filename = os.path.join(output_dir, f"elite_{gen}.csv")
        genome.Genome.to_csv(elite_dna, filename)

        # Track all-time best
        if max_fit > all_time_best_fitness:
            all_time_best_fitness = max_fit
            all_time_best_dna = elite_dna
            best_filename = os.path.join(output_dir, "best_ever.csv")
            genome.Genome.to_csv(elite_dna, best_filename)
            print(f"       -> New all-time best: {max_fit:.3f}")

        # 5. Selection and breeding (skip on last generation)
        if gen < generations - 1:
            # Handle zero fitness edge case
            if np.sum(fits) == 0:
                fits = [1.0 for _ in fits]  # Equal selection probability

            fit_map = population.Population.get_fitness_map(fits)
            new_creatures = []

            # Elitism: Keep the best creature unchanged
            elite_cr = creature.Creature(1)
            elite_cr.update_dna(elite_dna)
            new_creatures.append(elite_cr)

            # Breed the rest of the population
            while len(new_creatures) < len(pop.creatures):
                # Select parents via roulette wheel
                p1_ind = population.Population.select_parent(fit_map)
                p2_ind = population.Population.select_parent(fit_map)
                p1 = pop.creatures[p1_ind]
                p2 = pop.creatures[p2_ind]

                # Crossover
                dna = genome.Genome.crossover(p1.dna, p2.dna)

                # Mutations
                dna = genome.Genome.point_mutate(dna, rate=mutation_rate, amount=mutation_amount)
                dna = genome.Genome.shrink_mutate(dna, rate=shrink_rate)
                dna = genome.Genome.grow_mutate(dna, rate=grow_rate)

                # Create new creature with mutated DNA
                cr_new = creature.Creature(1)
                cr_new.update_dna(dna)
                new_creatures.append(cr_new)

            pop.creatures = new_creatures

    print("-" * 60)
    print(f"Evolution complete!")
    print(f"All-time best fitness: {all_time_best_fitness:.3f}")
    print(f"Best DNA saved to: {os.path.join(output_dir, 'best_ever.csv')}")

    return all_time_best_fitness, all_time_best_dna


if __name__ == "__main__":
    # Default settings - adjust these for experiments
    run_evolution(
        pop_size=20,        # Population size
        gene_count=5,       # Creature complexity
        generations=50,     # Number of generations
        sim_steps=2400,     # 10 seconds per simulation (240 fps)
        pool_size=4,        # Parallel workers (adjust to CPU cores)
        mutation_rate=0.1,
        mutation_amount=0.25,
        shrink_rate=0.25,
        grow_rate=0.1,
        output_dir="evolution_output"
    )
