"""
Experiment Runner for Mountain Climbing Evolution

This script runs multiple evolutionary experiments with different configurations
and compares their results. Experiments include:

1. Standard Evolution (baseline)
2. Snake Morphology (constrained body plan)
3. Heavy/Blocky Morphology
4. Control-Only Evolution (fixed body, evolve brain)
5. Sensor-Enabled Evolution (creatures sense mountain direction)

Usage:
    python run_experiments.py                    # Run all experiments
    python run_experiments.py --experiment snake # Run specific experiment
    python run_experiments.py --list             # List available experiments
"""

import argparse
import os
import json
import numpy as np
from datetime import datetime

import population
import genome
import creature
from genome_experiments import (
    SnakeGenome, HeavyGenome, ControlOnlyGenome, LeggedGenome
)
from mountain_simulation import MountainThreadedSim, SensorMountainThreadedSim


class ExperimentRunner:
    """Runs and tracks evolutionary experiments."""

    def __init__(self, output_base="experiments"):
        self.output_base = output_base
        self.results = {}

    def run_experiment(
        self,
        name,
        genome_class,
        sim_class,
        pop_size=20,
        gene_count=5,
        generations=30,
        sim_steps=2400,
        pool_size=4,
        mutation_rate=0.1,
        mutation_amount=0.25,
        shrink_rate=0.25,
        grow_rate=0.1
    ):
        """
        Run a single evolutionary experiment.

        Returns:
            dict: Results including fitness history and best DNA
        """
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_base, f"{name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {name}")
        print(f"{'='*60}")
        print(f"Genome: {genome_class.__name__}")
        print(f"Simulation: {sim_class.__name__}")
        print(f"Population: {pop_size}, Genes: {gene_count}, Generations: {generations}")
        print(f"Output: {output_dir}")
        print("-" * 60)

        # Initialize population with custom genome spec
        pop = population.Population(pop_size=pop_size, gene_count=gene_count)

        # If using custom genome, update creatures with new spec
        if genome_class != genome.Genome:
            for cr in pop.creatures:
                cr.spec = genome_class.get_gene_spec()
                cr.dna = genome_class.get_random_genome(len(cr.spec), gene_count)

        # Initialize simulation
        sim = sim_class(pool_size=pool_size)

        # Track results
        fitness_history = []
        best_fitness_history = []
        all_time_best = 0
        all_time_best_dna = None

        # Evolution loop
        for gen in range(generations):
            # Evaluate
            sim.eval_population(pop, sim_steps)

            # Collect stats
            fits = [cr.fitness for cr in pop.creatures]
            max_fit = np.max(fits)
            mean_fit = np.mean(fits)
            std_fit = np.std(fits)

            fitness_history.append({
                'generation': gen,
                'max': float(max_fit),
                'mean': float(mean_fit),
                'std': float(std_fit)
            })
            best_fitness_history.append(float(max_fit))

            print(f"Gen {gen:3d}: Max: {max_fit:.3f}, Mean: {mean_fit:.3f}, Std: {std_fit:.3f}")

            # Track best
            best_idx = np.argmax(fits)
            elite_dna = pop.creatures[best_idx].dna
            genome.Genome.to_csv(elite_dna, os.path.join(output_dir, f"elite_{gen}.csv"))

            if max_fit > all_time_best:
                all_time_best = max_fit
                all_time_best_dna = elite_dna
                genome.Genome.to_csv(elite_dna, os.path.join(output_dir, "best_ever.csv"))

            # Selection and breeding (skip last generation)
            if gen < generations - 1:
                if np.sum(fits) == 0:
                    fits = [1.0 for _ in fits]

                fit_map = population.Population.get_fitness_map(fits)
                new_creatures = []

                # Elitism
                elite_cr = creature.Creature(1)
                elite_cr.spec = genome_class.get_gene_spec()
                elite_cr.update_dna(elite_dna)
                new_creatures.append(elite_cr)

                # Breed
                while len(new_creatures) < len(pop.creatures):
                    p1_ind = population.Population.select_parent(fit_map)
                    p2_ind = population.Population.select_parent(fit_map)
                    p1 = pop.creatures[p1_ind]
                    p2 = pop.creatures[p2_ind]

                    # Use genome class methods for genetic operators
                    dna = genome_class.crossover(p1.dna, p2.dna)
                    dna = genome_class.point_mutate(dna, rate=mutation_rate, amount=mutation_amount)
                    dna = genome_class.shrink_mutate(dna, rate=shrink_rate)
                    dna = genome_class.grow_mutate(dna, rate=grow_rate)

                    cr_new = creature.Creature(1)
                    cr_new.spec = genome_class.get_gene_spec()
                    cr_new.update_dna(dna)
                    new_creatures.append(cr_new)

                pop.creatures = new_creatures

        # Save results
        results = {
            'experiment': name,
            'genome_class': genome_class.__name__,
            'sim_class': sim_class.__name__,
            'config': {
                'pop_size': pop_size,
                'gene_count': gene_count,
                'generations': generations,
                'sim_steps': sim_steps,
                'mutation_rate': mutation_rate
            },
            'best_fitness': float(all_time_best),
            'fitness_history': fitness_history,
            'output_dir': output_dir
        }

        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        print("-" * 60)
        print(f"Best fitness: {all_time_best:.3f}")
        print(f"Results saved to: {output_dir}")

        self.results[name] = results
        return results


def get_experiment_configs():
    """Define available experiments."""
    return {
        'standard': {
            'name': 'standard',
            'genome_class': genome.Genome,
            'sim_class': MountainThreadedSim,
            'description': 'Baseline evolution with standard genome'
        },
        'snake': {
            'name': 'snake',
            'genome_class': SnakeGenome,
            'sim_class': MountainThreadedSim,
            'description': 'Snake-like morphology (thin, chain-like bodies)'
        },
        'heavy': {
            'name': 'heavy',
            'genome_class': HeavyGenome,
            'sim_class': MountainThreadedSim,
            'description': 'Heavy/blocky morphology (stable base)'
        },
        'control_only': {
            'name': 'control_only',
            'genome_class': ControlOnlyGenome,
            'sim_class': MountainThreadedSim,
            'description': 'Fixed morphology, only evolve motor control'
        },
        'legged': {
            'name': 'legged',
            'genome_class': LeggedGenome,
            'sim_class': MountainThreadedSim,
            'description': 'Legged morphology optimized for walking'
        },
        'sensor': {
            'name': 'sensor',
            'genome_class': genome.Genome,
            'sim_class': SensorMountainThreadedSim,
            'description': 'Standard genome with sensory feedback'
        },
        'sensor_snake': {
            'name': 'sensor_snake',
            'genome_class': SnakeGenome,
            'sim_class': SensorMountainThreadedSim,
            'description': 'Snake morphology with sensory feedback'
        }
    }


def print_comparison(results):
    """Print comparison of experiment results."""
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPARISON")
    print("=" * 60)
    print(f"{'Experiment':<20} {'Best Fitness':>12} {'Final Mean':>12}")
    print("-" * 60)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['best_fitness'], reverse=True)

    for name, result in sorted_results:
        final_mean = result['fitness_history'][-1]['mean']
        print(f"{name:<20} {result['best_fitness']:>12.3f} {final_mean:>12.3f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Run evolutionary experiments')
    parser.add_argument('--experiment', '-e', type=str, help='Specific experiment to run')
    parser.add_argument('--list', '-l', action='store_true', help='List available experiments')
    parser.add_argument('--generations', '-g', type=int, default=30, help='Number of generations')
    parser.add_argument('--population', '-p', type=int, default=20, help='Population size')
    parser.add_argument('--genes', type=int, default=5, help='Gene count per creature')
    parser.add_argument('--pool', type=int, default=4, help='Parallel worker pool size')
    parser.add_argument('--output', '-o', type=str, default='experiments', help='Output directory')

    args = parser.parse_args()

    experiments = get_experiment_configs()

    if args.list:
        print("\nAvailable Experiments:")
        print("-" * 60)
        for name, config in experiments.items():
            print(f"  {name:<15} - {config['description']}")
        print()
        return

    runner = ExperimentRunner(output_base=args.output)

    if args.experiment:
        # Run specific experiment
        if args.experiment not in experiments:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {', '.join(experiments.keys())}")
            return

        config = experiments[args.experiment]
        runner.run_experiment(
            name=config['name'],
            genome_class=config['genome_class'],
            sim_class=config['sim_class'],
            pop_size=args.population,
            gene_count=args.genes,
            generations=args.generations,
            pool_size=args.pool
        )
    else:
        # Run all experiments
        print("Running all experiments...")
        for name, config in experiments.items():
            runner.run_experiment(
                name=config['name'],
                genome_class=config['genome_class'],
                sim_class=config['sim_class'],
                pop_size=args.population,
                gene_count=args.genes,
                generations=args.generations,
                pool_size=args.pool
            )

        # Print comparison
        print_comparison(runner.results)


if __name__ == "__main__":
    main()
