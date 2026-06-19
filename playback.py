"""
Playback Script for Visualizing Evolved Creatures

This script loads a saved creature DNA and runs it in GUI mode
for visualization and video recording.

Usage:
    python playback.py                                    # Uses default best
    python playback.py --file experiment_results/baseline_pop50_best.csv
    python playback.py --experiment baseline_pop50 --gen 0   # Gen 0 (early)
    python playback.py --experiment baseline_pop50 --gen 49  # Gen 49 (evolved)
    python playback.py --experiment snake_morphology         # Snake creature
    python playback.py --file best.csv --landscape steep_mountain.urdf

Video Recording Workflow:
    Shot A (The Idiot):     --experiment baseline_pop50 --gen 0
    Shot B (The Pro):       --experiment baseline_pop50 --gen 49
    Shot C (The Specialist): --experiment snake_morphology
    Shot D (The Smart One): --experiment sensor_creatures
"""

import pybullet as p
import pybullet_data
import time
import sys
import os
import argparse
import creature
import genome


def build_arena(pid, arena_size=20, wall_height=1, wall_thickness=0.5):
    """Build the arena with floor and walls."""
    # Floor
    floor_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[arena_size/2, arena_size/2, wall_thickness],
        physicsClientId=pid
    )
    floor_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[arena_size/2, arena_size/2, wall_thickness],
        rgbaColor=[1, 1, 0, 1],
        physicsClientId=pid
    )
    p.createMultiBody(0, floor_col, floor_vis, [0, 0, -wall_thickness], physicsClientId=pid)

    # Walls (North/South)
    wall_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[arena_size/2, wall_thickness/2, wall_height/2],
        physicsClientId=pid
    )
    wall_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[arena_size/2, wall_thickness/2, wall_height/2],
        rgbaColor=[0.7, 0.7, 0.7, 1],
        physicsClientId=pid
    )
    p.createMultiBody(0, wall_col, wall_vis, [0, arena_size/2, wall_height/2], physicsClientId=pid)
    p.createMultiBody(0, wall_col, wall_vis, [0, -arena_size/2, wall_height/2], physicsClientId=pid)

    # Walls (East/West)
    wall_col_side = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[wall_thickness/2, arena_size/2, wall_height/2],
        physicsClientId=pid
    )
    wall_vis_side = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[wall_thickness/2, arena_size/2, wall_height/2],
        rgbaColor=[0.7, 0.7, 0.7, 1],
        physicsClientId=pid
    )
    p.createMultiBody(0, wall_col_side, wall_vis_side, [arena_size/2, 0, wall_height/2], physicsClientId=pid)
    p.createMultiBody(0, wall_col_side, wall_vis_side, [-arena_size/2, 0, wall_height/2], physicsClientId=pid)


def run_playback(dna_file, landscape="gaussian_pyramid.urdf", iterations=2400, realtime=True, title=None):
    """
    Run a creature in GUI mode for visualization.

    Args:
        dna_file: Path to CSV file containing creature DNA
        landscape: URDF file for the mountain landscape
        iterations: Number of simulation steps
        realtime: If True, run at real-time speed (240 fps)
        title: Optional title to display
    """
    # Load DNA
    print(f"\n{'='*60}")
    if title:
        print(f"PLAYBACK: {title}")
    print(f"DNA File: {dna_file}")
    print(f"Landscape: {landscape}")
    print(f"{'='*60}")

    dna = genome.Genome.from_csv(dna_file)

    # Create creature from DNA
    cr = creature.Creature(1)
    cr.update_dna(dna)
    print(f"Creature has {len(cr.get_expanded_links())} links")

    # Connect to PyBullet GUI
    pid = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=pid)  # Hide debug panels
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=pid)
    p.setGravity(0, 0, -10, physicsClientId=pid)

    # Set up camera for good viewing angle
    p.resetDebugVisualizerCamera(
        cameraDistance=15,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 2],
        physicsClientId=pid
    )

    # Build arena
    build_arena(pid)

    # Load mountain (configurable landscape)
    p.setAdditionalSearchPath('shapes/', physicsClientId=pid)
    mountain_pos = [0, 0, -1]
    mountain_orn = [0, 0, 0, 1]
    try:
        p.loadURDF(landscape, mountain_pos, mountain_orn, useFixedBase=1, physicsClientId=pid)
    except Exception as e:
        print(f"Error loading mountain: {e}")
        print(f"Make sure 'shapes/' directory contains {landscape}")
        p.disconnect(pid)
        return

    # Save creature to URDF and load
    xml_file = 'playback_creature.urdf'
    with open(xml_file, 'w') as f:
        f.write(cr.to_xml())

    start_pos = [0, -8, 1.5]
    cid = p.loadURDF(xml_file, start_pos, physicsClientId=pid)

    # Run simulation
    print(f"\nRunning simulation for {iterations} steps ({iterations/240:.1f} seconds)")
    print("Press Ctrl+C to exit early")

    max_height = 0
    try:
        for step in range(iterations):
            p.stepSimulation(physicsClientId=pid)

            # Update motors
            if step % 24 == 0:
                for jid in range(p.getNumJoints(cid, physicsClientId=pid)):
                    m = cr.get_motors()[jid]
                    p.setJointMotorControl2(
                        cid, jid,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=m.get_output(),
                        force=5,
                        physicsClientId=pid
                    )

            # Track height
            pos, _ = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
            if pos[2] > max_height:
                max_height = pos[2]

            # Real-time delay
            if realtime:
                time.sleep(1/240)

            # Print progress every second
            if step % 240 == 0:
                print(f"  Step {step}/{iterations}, Height: {pos[2]:.3f}, Max: {max_height:.3f}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted")

    print(f"\nFinal max height achieved: {max_height:.3f}")
    print(f"Climbing fitness: {max(0, max_height - 1.5):.3f}")

    # Keep window open
    print("\nPress Enter to close...")
    input()
    p.disconnect(pid)


def find_dna_file(experiment, gen, results_dir):
    """
    Find the DNA file for a given experiment and generation.

    Args:
        experiment: Experiment name (e.g., 'baseline_pop50')
        gen: Generation number (None for best overall)
        results_dir: Directory containing experiment results

    Returns:
        Path to DNA file, or None if not found
    """
    if gen is not None:
        # Look for experiment-specific elite files (primary format)
        exp_elite_path = os.path.join(results_dir, f"{experiment}_elite_{gen}.csv")
        if os.path.exists(exp_elite_path):
            return exp_elite_path

        # Fallback: results_dir/elite_{gen}.csv (flat structure)
        flat_path = os.path.join(results_dir, f"elite_{gen}.csv")
        if os.path.exists(flat_path):
            return flat_path

        # Fallback: results_dir/{experiment}/elite_{gen}.csv
        nested_path = os.path.join(results_dir, experiment, f"elite_{gen}.csv")
        if os.path.exists(nested_path):
            return nested_path

        print(f"Warning: Could not find elite for {experiment} gen {gen}")
        print(f"  Tried: {exp_elite_path}")
        print(f"  Tried: {flat_path}")
        return None

    else:
        # Look for best overall
        best_path = os.path.join(results_dir, f"{experiment}_best.csv")
        if os.path.exists(best_path):
            return best_path

        # Fallback to best_ever.csv
        best_ever = os.path.join(results_dir, "best_ever.csv")
        if os.path.exists(best_ever):
            return best_ever

        return None


def main():
    parser = argparse.ArgumentParser(
        description='Visualize evolved creatures in PyBullet GUI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python playback.py --experiment baseline_pop50 --gen 0    # Early generation
  python playback.py --experiment baseline_pop50 --gen 49   # Final generation
  python playback.py --experiment snake_morphology          # Snake creature
  python playback.py --experiment sensor_creatures          # Sensor creature
  python playback.py --file my_creature.csv                 # Specific file
  python playback.py --file best.csv --landscape steep_mountain.urdf

Video Shot List:
  Shot A (The Idiot):      --experiment baseline_pop50 --gen 0
  Shot B (The Pro):        --experiment baseline_pop50 --gen 49
  Shot C (The Specialist): --experiment snake_morphology
  Shot D (The Smart One):  --experiment sensor_creatures
        """
    )

    parser.add_argument('--file', '-f', type=str,
                        help='Direct path to DNA CSV file')
    parser.add_argument('--experiment', '-e', type=str,
                        help='Experiment name (e.g., baseline_pop50, snake_morphology)')
    parser.add_argument('--gen', '-g', type=int, default=None,
                        help='Generation number (default: best overall)')
    parser.add_argument('--landscape', '-l', type=str, default='gaussian_pyramid.urdf',
                        help='Landscape URDF file (default: gaussian_pyramid.urdf)')
    parser.add_argument('--results', '-r', type=str, default='experiment_results',
                        help='Results directory (default: experiment_results)')
    parser.add_argument('--fast', action='store_true',
                        help='Run at maximum speed (not real-time)')
    parser.add_argument('--duration', '-d', type=int, default=2400,
                        help='Simulation duration in steps (default: 2400 = 10 seconds)')

    args = parser.parse_args()

    # Determine DNA file
    dna_file = None
    title = None

    if args.file:
        dna_file = args.file
        title = os.path.basename(args.file)
    elif args.experiment:
        dna_file = find_dna_file(args.experiment, args.gen, args.results)
        if args.gen is not None:
            title = f"{args.experiment} - Generation {args.gen}"
        else:
            title = f"{args.experiment} - Best"
    else:
        # Default: look for best_ever.csv
        default_paths = [
            "experiment_results/best_ever.csv",
            "evolution_output/best_ever.csv",
            "best_ever.csv"
        ]
        for path in default_paths:
            if os.path.exists(path):
                dna_file = path
                title = "Best Overall"
                break

    if not dna_file or not os.path.exists(dna_file):
        print("Error: Could not find DNA file")
        print("\nUsage examples:")
        print("  python playback.py --experiment baseline_pop50 --gen 49")
        print("  python playback.py --file experiment_results/baseline_pop50_best.csv")
        print("\nRun collect_experiment_data.py first to generate creature DNA")
        sys.exit(1)

    run_playback(
        dna_file=dna_file,
        landscape=args.landscape,
        iterations=args.duration,
        realtime=not args.fast,
        title=title
    )


if __name__ == "__main__":
    main()
