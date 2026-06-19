"""
Graph Generation Script for Coursework Report

This script reads experiment CSV files and generates publication-quality
graphs for the report.

Required Graphs:
1. Fitness vs Generation: Baseline vs High Population
2. Fitness vs Generation: Standard Body vs Snake Body
3. Bar Chart: Blind vs Sensor-equipped creatures (Max Height)

Additional Graphs:
4. Fitness vs Generation: All tuning experiments
5. Fitness vs Generation: All encoding experiments
6. Summary comparison bar chart

Usage:
    python generate_graphs.py                    # Generate all graphs
    python generate_graphs.py --input results/  # Use custom input directory
"""

import os
import csv
import json
import argparse

# Try to import matplotlib, provide instructions if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches  # noqa: F401  (availability check)
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")

try:
    import numpy as np  # noqa: F401  (availability check)
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def load_fitness_csv(filepath):
    """Load fitness data from CSV file."""
    generations = []
    max_fitness = []
    mean_fitness = []
    std_fitness = []

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            generations.append(int(row['generation']))
            max_fitness.append(float(row['max_fitness']))
            mean_fitness.append(float(row['mean_fitness']))
            std_fitness.append(float(row['std_fitness']))

    return {
        'generations': generations,
        'max_fitness': max_fitness,
        'mean_fitness': mean_fitness,
        'std_fitness': std_fitness
    }


def graph1_baseline_vs_highpop(input_dir, output_dir):
    """
    Graph 1: Fitness vs Generation - Baseline vs High Population
    Shows impact of population size on evolution speed.
    """
    baseline_file = os.path.join(input_dir, "baseline_pop50.csv")
    highpop_file = os.path.join(input_dir, "high_pop100.csv")

    if not os.path.exists(baseline_file) or not os.path.exists(highpop_file):
        print("Graph 1: Missing data files, skipping...")
        return

    baseline = load_fitness_csv(baseline_file)
    highpop = load_fitness_csv(highpop_file)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot max fitness
    ax.plot(baseline['generations'], baseline['max_fitness'],
            'b-', linewidth=2, label='Baseline (Pop=50) - Max')
    ax.plot(highpop['generations'], highpop['max_fitness'],
            'r-', linewidth=2, label='High Pop (Pop=100) - Max')

    # Plot mean fitness with shaded std
    ax.plot(baseline['generations'], baseline['mean_fitness'],
            'b--', linewidth=1, alpha=0.7, label='Baseline - Mean')
    ax.plot(highpop['generations'], highpop['mean_fitness'],
            'r--', linewidth=1, alpha=0.7, label='High Pop - Mean')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness (Max Height)', fontsize=12)
    ax.set_title('Effect of Population Size on Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(baseline['generations']))
    ax.set_ylim(0, None)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "graph1_population_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def graph2_standard_vs_snake(input_dir, output_dir):
    """
    Graph 2: Fitness vs Generation - Standard Body vs Snake Body
    Shows impact of morphology constraints on climbing ability.
    """
    standard_file = os.path.join(input_dir, "baseline_pop50.csv")
    snake_file = os.path.join(input_dir, "snake_morphology.csv")

    if not os.path.exists(standard_file) or not os.path.exists(snake_file):
        print("Graph 2: Missing data files, skipping...")
        return

    standard = load_fitness_csv(standard_file)
    snake = load_fitness_csv(snake_file)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot max fitness
    ax.plot(standard['generations'], standard['max_fitness'],
            'b-', linewidth=2, label='Standard Genome - Max')
    ax.plot(snake['generations'], snake['max_fitness'],
            'g-', linewidth=2, label='Snake Morphology - Max')

    # Plot mean fitness
    ax.plot(standard['generations'], standard['mean_fitness'],
            'b--', linewidth=1, alpha=0.7, label='Standard - Mean')
    ax.plot(snake['generations'], snake['mean_fitness'],
            'g--', linewidth=1, alpha=0.7, label='Snake - Mean')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness (Max Height)', fontsize=12)
    ax.set_title('Standard vs Snake Morphology', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(standard['generations']))
    ax.set_ylim(0, None)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "graph2_morphology_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def graph3_blind_vs_sensor(input_dir, output_dir):
    """
    Graph 3: Bar Chart - Blind vs Sensor-equipped creatures
    Shows impact of sensory feedback on climbing performance.
    """
    summary_file = os.path.join(input_dir, "experiment_summary.json")

    if not os.path.exists(summary_file):
        # Try loading from individual files
        blind_file = os.path.join(input_dir, "blind_creatures.csv")
        sensor_file = os.path.join(input_dir, "sensor_creatures.csv")

        if not os.path.exists(blind_file) or not os.path.exists(sensor_file):
            print("Graph 3: Missing data files, skipping...")
            return

        blind = load_fitness_csv(blind_file)
        sensor = load_fitness_csv(sensor_file)

        blind_best = max(blind['max_fitness'])
        sensor_best = max(sensor['max_fitness'])
        blind_mean = blind['mean_fitness'][-1]
        sensor_mean = sensor['mean_fitness'][-1]
    else:
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        blind_best = summary.get('blind_creatures', {}).get('best', 0)
        sensor_best = summary.get('sensor_creatures', {}).get('best', 0)
        blind_mean = summary.get('blind_creatures', {}).get('final_mean', 0)
        sensor_mean = summary.get('sensor_creatures', {}).get('final_mean', 0)

    fig, ax = plt.subplots(figsize=(8, 6))

    x = [0, 1]
    width = 0.35

    # Best fitness bars
    best_bars = ax.bar([i - width/2 for i in x], [blind_best, sensor_best],
                       width, label='Best Fitness', color=['#3498db', '#e74c3c'])

    # Mean fitness bars
    mean_bars = ax.bar([i + width/2 for i in x], [blind_mean, sensor_mean],
                       width, label='Final Mean Fitness', color=['#85c1e9', '#f1948a'])

    ax.set_ylabel('Fitness (Max Height)', fontsize=12)
    ax.set_title('Blind vs Sensor-Equipped Creatures', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Blind', 'Sensor-Equipped'], fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in best_bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    for bar in mean_bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "graph3_blind_vs_sensor.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def graph4_all_tuning(input_dir, output_dir):
    """
    Graph 4: All tuning experiments comparison
    """
    experiments = [
        ("baseline_pop50.csv", "Baseline (Pop=50)", "blue"),
        ("high_pop100.csv", "High Pop (Pop=100)", "red"),
        ("high_mutation.csv", "High Mutation (0.2)", "green"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for filename, label, color in experiments:
        filepath = os.path.join(input_dir, filename)
        if os.path.exists(filepath):
            data = load_fitness_csv(filepath)
            ax.plot(data['generations'], data['max_fitness'],
                    color=color, linewidth=2, label=f'{label} - Max')
            ax.plot(data['generations'], data['mean_fitness'],
                    color=color, linewidth=1, linestyle='--', alpha=0.7, label=f'{label} - Mean')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness (Max Height)', fontsize=12)
    ax.set_title('GA Parameter Tuning Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "graph4_tuning_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def graph5_encoding_experiments(input_dir, output_dir):
    """
    Graph 5: All encoding experiments comparison
    """
    experiments = [
        ("baseline_pop50.csv", "Standard Genome", "blue"),
        ("snake_morphology.csv", "Snake Morphology", "green"),
        ("control_only.csv", "Control-Only", "orange"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for filename, label, color in experiments:
        filepath = os.path.join(input_dir, filename)
        if os.path.exists(filepath):
            data = load_fitness_csv(filepath)
            ax.plot(data['generations'], data['max_fitness'],
                    color=color, linewidth=2, label=f'{label} - Max')
            ax.plot(data['generations'], data['mean_fitness'],
                    color=color, linewidth=1, linestyle='--', alpha=0.7, label=f'{label} - Mean')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness (Max Height)', fontsize=12)
    ax.set_title('Encoding Scheme Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "graph5_encoding_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def graph6_summary_bar(input_dir, output_dir):
    """
    Graph 6: Summary bar chart of all experiments
    """
    summary_file = os.path.join(input_dir, "experiment_summary.json")

    if not os.path.exists(summary_file):
        print("Graph 6: Missing summary file, skipping...")
        return

    with open(summary_file, 'r') as f:
        summary = json.load(f)

    # Prepare data
    experiments = list(summary.keys())
    best_fitness = [summary[exp]['best'] for exp in experiments]
    final_mean = [summary[exp]['final_mean'] for exp in experiments]

    # Shorten labels
    labels = [exp.replace('_', '\n') for exp in experiments]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(experiments))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], best_fitness, width,
                   label='Best Fitness', color='#3498db')
    bars2 = ax.bar([i + width/2 for i in x], final_mean, width,
                   label='Final Mean', color='#2ecc71')

    ax.set_ylabel('Fitness (Max Height)', fontsize=12)
    ax.set_title('Summary: All Experiments', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = os.path.join(output_dir, "graph6_summary.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def graph7_sensor_evolution(input_dir, output_dir):
    """
    Graph 7: Blind vs Sensor fitness over generations
    """
    blind_file = os.path.join(input_dir, "blind_creatures.csv")
    sensor_file = os.path.join(input_dir, "sensor_creatures.csv")

    if not os.path.exists(blind_file) or not os.path.exists(sensor_file):
        print("Graph 7: Missing data files, skipping...")
        return

    blind = load_fitness_csv(blind_file)
    sensor = load_fitness_csv(sensor_file)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(blind['generations'], blind['max_fitness'],
            'b-', linewidth=2, label='Blind - Max')
    ax.plot(sensor['generations'], sensor['max_fitness'],
            'r-', linewidth=2, label='Sensor - Max')

    ax.fill_between(blind['generations'],
                    [m - s for m, s in zip(blind['mean_fitness'], blind['std_fitness'])],
                    [m + s for m, s in zip(blind['mean_fitness'], blind['std_fitness'])],
                    alpha=0.2, color='blue', label='Blind Std Dev')

    ax.fill_between(sensor['generations'],
                    [m - s for m, s in zip(sensor['mean_fitness'], sensor['std_fitness'])],
                    [m + s for m, s in zip(sensor['mean_fitness'], sensor['std_fitness'])],
                    alpha=0.2, color='red', label='Sensor Std Dev')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness (Max Height)', fontsize=12)
    ax.set_title('Blind vs Sensor-Equipped Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "graph7_sensor_evolution.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def graph8_landscape_comparison(input_dir, output_dir):
    """
    Graph 8: Landscape Generalization (Exceptional Criteria)
    Compares evolution on standard vs steep mountain.
    """
    baseline_file = os.path.join(input_dir, "baseline_pop50.csv")
    steep_file = os.path.join(input_dir, "steep_landscape.csv")

    if not os.path.exists(baseline_file) or not os.path.exists(steep_file):
        print("Graph 8: Missing landscape data files, skipping...")
        return

    baseline = load_fitness_csv(baseline_file)
    steep = load_fitness_csv(steep_file)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot max fitness
    ax.plot(baseline['generations'], baseline['max_fitness'],
            'b-', linewidth=2, label='Standard Mountain - Max')
    ax.plot(steep['generations'], steep['max_fitness'],
            'purple', linewidth=2, label='Steep Mountain - Max')

    # Plot mean fitness
    ax.plot(baseline['generations'], baseline['mean_fitness'],
            'b--', linewidth=1, alpha=0.7, label='Standard - Mean')
    ax.plot(steep['generations'], steep['mean_fitness'],
            color='purple', linewidth=1, linestyle='--', alpha=0.7, label='Steep - Mean')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness (Max Height)', fontsize=12)
    ax.set_title('Landscape Generalization: Standard vs Steep Mountain', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(baseline['generations']))
    ax.set_ylim(0, None)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "graph8_landscape_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def generate_all_graphs(input_dir, output_dir):
    """Generate all graphs for the report."""
    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating graphs...")
    print("="*50)

    # Required graphs (as per brief)
    graph1_baseline_vs_highpop(input_dir, output_dir)
    graph2_standard_vs_snake(input_dir, output_dir)
    graph3_blind_vs_sensor(input_dir, output_dir)

    # Additional graphs for comprehensive analysis
    graph4_all_tuning(input_dir, output_dir)
    graph5_encoding_experiments(input_dir, output_dir)
    graph6_summary_bar(input_dir, output_dir)
    graph7_sensor_evolution(input_dir, output_dir)

    # Exceptional criteria: Landscape generalization
    graph8_landscape_comparison(input_dir, output_dir)

    print("\n" + "="*50)
    print(f"All graphs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate graphs for coursework report')
    parser.add_argument('--input', '-i', type=str, default='experiment_results',
                        help='Input directory with CSV files')
    parser.add_argument('--output', '-o', type=str, default='graphs',
                        help='Output directory for graphs')

    args = parser.parse_args()

    generate_all_graphs(args.input, args.output)


if __name__ == "__main__":
    main()
