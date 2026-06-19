"""
Experiment Data Collection Script

This script runs all required experiments for the coursework report and
saves the results to CSV files for analysis and graph generation.

Experiments:
1. Baseline (Pop=50, standard genome)
2. High Population (Pop=100)
3. High Mutation Rate (rate=0.2)
4. Snake Morphology
5. Control-Only Evolution
6. Sensor-Enabled Evolution
7. Blind vs Sensor Comparison

Output:
- CSV files with fitness data per generation
- Summary statistics for each experiment
"""

import os
import csv
import json
import gc
import numpy as np

import population
import genome
import creature
from genome_experiments import SnakeGenome, ControlOnlyGenome
from mountain_simulation import MountainThreadedSim, SensorMountainThreadedSim


def run_single_experiment(
    name,
    genome_class,
    sim_class,
    pop_size,
    gene_count,
    generations,
    sim_steps,
    pool_size,
    mutation_rate,
    output_dir,
    landscape="gaussian_pyramid.urdf"
):
    """
    Run a single experiment and return fitness history.
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"Pop: {pop_size}, Genes: {gene_count}, Gens: {generations}")
    print(f"Mutation Rate: {mutation_rate}")
    print(f"{'='*60}")

    # Initialize population
    pop = population.Population(pop_size=pop_size, gene_count=gene_count)

    # Apply custom genome if needed
    if genome_class != genome.Genome:
        for cr in pop.creatures:
            cr.spec = genome_class.get_gene_spec()
            cr.dna = genome_class.get_random_genome(len(cr.spec), gene_count)

    # Initialize simulation with landscape
    sim = sim_class(pool_size=pool_size, landscape=landscape)

    # Track results
    fitness_data = []
    all_time_best = 0
    all_time_best_dna = None

    for gen in range(generations):
        # Evaluate
        sim.eval_population(pop, sim_steps)

        # Collect stats
        fits = [cr.fitness for cr in pop.creatures]
        max_fit = np.max(fits)
        mean_fit = np.mean(fits)
        std_fit = np.std(fits)
        min_fit = np.min(fits)

        fitness_data.append({
            'generation': gen,
            'max_fitness': float(max_fit),
            'mean_fitness': float(mean_fit),
            'std_fitness': float(std_fit),
            'min_fitness': float(min_fit)
        })

        print(f"Gen {gen:3d}: Max={max_fit:.3f}, Mean={mean_fit:.3f}, Std={std_fit:.3f}")

        # Track best
        best_idx = np.argmax(fits)
        elite_dna = pop.creatures[best_idx].dna

        # Save elite DNA for this generation (for video workflow)
        elite_file = os.path.join(output_dir, f"{name}_elite_{gen}.csv")
        genome.Genome.to_csv(elite_dna, elite_file)

        if max_fit > all_time_best:
            all_time_best = max_fit
            all_time_best_dna = elite_dna

        # Selection and breeding
        if gen < generations - 1:
            if np.sum(fits) == 0:
                fits = [1.0 for _ in fits]

            fit_map = population.Population.get_fitness_map(fits)
            new_creatures = []

            # Elitism
            elite_cr = creature.Creature(1)
            if genome_class != genome.Genome:
                elite_cr.spec = genome_class.get_gene_spec()
            elite_cr.update_dna(elite_dna)
            new_creatures.append(elite_cr)

            # Breed
            while len(new_creatures) < len(pop.creatures):
                p1_ind = population.Population.select_parent(fit_map)
                p2_ind = population.Population.select_parent(fit_map)
                p1 = pop.creatures[p1_ind]
                p2 = pop.creatures[p2_ind]

                dna = genome_class.crossover(p1.dna, p2.dna)
                dna = genome_class.point_mutate(dna, rate=mutation_rate, amount=0.25)
                dna = genome_class.shrink_mutate(dna, rate=0.25)
                dna = genome_class.grow_mutate(dna, rate=0.1)

                cr_new = creature.Creature(1)
                if genome_class != genome.Genome:
                    cr_new.spec = genome_class.get_gene_spec()
                cr_new.update_dna(dna)
                new_creatures.append(cr_new)

            pop.creatures = new_creatures

    # Save best DNA
    if all_time_best_dna is not None:
        dna_file = os.path.join(output_dir, f"{name}_best.csv")
        genome.Genome.to_csv(all_time_best_dna, dna_file)

    print(f"Best fitness achieved: {all_time_best:.3f}")

    # Clean up memory between experiments
    gc.collect()

    return fitness_data, all_time_best


def save_fitness_csv(fitness_data, filename):
    """Save fitness data to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['generation', 'max_fitness', 'mean_fitness', 'std_fitness', 'min_fitness'])
        writer.writeheader()
        writer.writerows(fitness_data)
    print(f"Saved: {filename}")


def run_all_experiments(output_dir="experiment_results", generations=30, pool_size=4):
    """
    Run all required experiments for the coursework.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Store summary results
    summary = {}

    # ========================================
    # GOAL 1: Baseline Run
    # ========================================
    print("\n" + "#"*60)
    print("# GOAL 1: BASELINE RUN")
    print("#"*60)

    data, best = run_single_experiment(
        name="baseline_pop50",
        genome_class=genome.Genome,
        sim_class=MountainThreadedSim,
        pop_size=50,
        gene_count=5,
        generations=generations,
        sim_steps=2400,
        pool_size=pool_size,
        mutation_rate=0.1,
        output_dir=output_dir
    )
    save_fitness_csv(data, os.path.join(output_dir, "baseline_pop50.csv"))
    summary["baseline_pop50"] = {"best": best, "final_mean": data[-1]['mean_fitness']}

    # ========================================
    # GOAL 2: Tuning Experiments
    # ========================================
    print("\n" + "#"*60)
    print("# GOAL 2: TUNING EXPERIMENTS")
    print("#"*60)

    # High Population (Pop=100)
    data, best = run_single_experiment(
        name="high_pop100",
        genome_class=genome.Genome,
        sim_class=MountainThreadedSim,
        pop_size=100,
        gene_count=5,
        generations=generations,
        sim_steps=2400,
        pool_size=pool_size,
        mutation_rate=0.1,
        output_dir=output_dir
    )
    save_fitness_csv(data, os.path.join(output_dir, "high_pop100.csv"))
    summary["high_pop100"] = {"best": best, "final_mean": data[-1]['mean_fitness']}

    # High Mutation Rate (0.2)
    data, best = run_single_experiment(
        name="high_mutation",
        genome_class=genome.Genome,
        sim_class=MountainThreadedSim,
        pop_size=50,
        gene_count=5,
        generations=generations,
        sim_steps=2400,
        pool_size=pool_size,
        mutation_rate=0.2,
        output_dir=output_dir
    )
    save_fitness_csv(data, os.path.join(output_dir, "high_mutation.csv"))
    summary["high_mutation"] = {"best": best, "final_mean": data[-1]['mean_fitness']}

    # ========================================
    # GOAL 3: Advanced Encoding Experiments
    # ========================================
    print("\n" + "#"*60)
    print("# GOAL 3: ENCODING EXPERIMENTS")
    print("#"*60)

    # Snake Morphology
    data, best = run_single_experiment(
        name="snake_morphology",
        genome_class=SnakeGenome,
        sim_class=MountainThreadedSim,
        pop_size=50,
        gene_count=5,
        generations=generations,
        sim_steps=2400,
        pool_size=pool_size,
        mutation_rate=0.1,
        output_dir=output_dir
    )
    save_fitness_csv(data, os.path.join(output_dir, "snake_morphology.csv"))
    summary["snake_morphology"] = {"best": best, "final_mean": data[-1]['mean_fitness']}

    # Control-Only Evolution
    data, best = run_single_experiment(
        name="control_only",
        genome_class=ControlOnlyGenome,
        sim_class=MountainThreadedSim,
        pop_size=50,
        gene_count=5,
        generations=generations,
        sim_steps=2400,
        pool_size=pool_size,
        mutation_rate=0.1,
        output_dir=output_dir
    )
    save_fitness_csv(data, os.path.join(output_dir, "control_only.csv"))
    summary["control_only"] = {"best": best, "final_mean": data[-1]['mean_fitness']}

    # ========================================
    # GOAL 4: Exceptional Experiments (Sensors)
    # ========================================
    print("\n" + "#"*60)
    print("# GOAL 4: SENSOR EXPERIMENTS")
    print("#"*60)

    # Blind creatures (baseline for comparison)
    data_blind, best_blind = run_single_experiment(
        name="blind_creatures",
        genome_class=genome.Genome,
        sim_class=MountainThreadedSim,
        pop_size=50,
        gene_count=5,
        generations=generations,
        sim_steps=2400,
        pool_size=pool_size,
        mutation_rate=0.1,
        output_dir=output_dir
    )
    save_fitness_csv(data_blind, os.path.join(output_dir, "blind_creatures.csv"))
    summary["blind_creatures"] = {"best": best_blind, "final_mean": data_blind[-1]['mean_fitness']}

    # Sensor-enabled creatures
    data_sensor, best_sensor = run_single_experiment(
        name="sensor_creatures",
        genome_class=genome.Genome,
        sim_class=SensorMountainThreadedSim,
        pop_size=50,
        gene_count=5,
        generations=generations,
        sim_steps=2400,
        pool_size=pool_size,
        mutation_rate=0.1,
        output_dir=output_dir
    )
    save_fitness_csv(data_sensor, os.path.join(output_dir, "sensor_creatures.csv"))
    summary["sensor_creatures"] = {"best": best_sensor, "final_mean": data_sensor[-1]['mean_fitness']}

    # ========================================
    # GOAL 5: Exceptional - Landscape Variation
    # ========================================
    print("\n" + "#"*60)
    print("# GOAL 5: LANDSCAPE EXPERIMENTS (EXCEPTIONAL)")
    print("#"*60)

    # Steep Mountain Landscape
    data, best = run_single_experiment(
        name="steep_landscape",
        genome_class=genome.Genome,
        sim_class=MountainThreadedSim,
        pop_size=50,
        gene_count=5,
        generations=generations,
        sim_steps=2400,
        pool_size=pool_size,
        mutation_rate=0.1,
        output_dir=output_dir,
        landscape="steep_mountain.urdf"
    )
    save_fitness_csv(data, os.path.join(output_dir, "steep_landscape.csv"))
    summary["steep_landscape"] = {"best": best, "final_mean": data[-1]['mean_fitness']}

    # ========================================
    # Save Summary
    # ========================================
    summary_file = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"\nSummary saved to: {summary_file}")
    print("\nResults:")
    print(f"{'Experiment':<20} {'Best Fitness':>12} {'Final Mean':>12}")
    print("-" * 50)
    for name, result in summary.items():
        print(f"{name:<20} {result['best']:>12.3f} {result['final_mean']:>12.3f}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run experiments and collect data')
    parser.add_argument('--generations', '-g', type=int, default=30, help='Generations per experiment')
    parser.add_argument('--pool', '-p', type=int, default=4, help='Worker pool size')
    parser.add_argument('--output', '-o', type=str, default='experiment_results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test run (10 generations)')

    args = parser.parse_args()

    gens = 10 if args.quick else args.generations

    run_all_experiments(
        output_dir=args.output,
        generations=gens,
        pool_size=args.pool
    )
