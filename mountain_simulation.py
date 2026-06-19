"""
Mountain Climbing Simulation for Evolutionary Creatures

This module provides simulation environments for evolving creatures to climb
a mountain. It uses a STABLE HEIGHT fitness function that rewards creatures
for actually climbing and staying on the mountain, not flying.

Key Design Decisions:
1. FITNESS = minimum height in final second (rewards stability, penalizes flying)
2. PENALTY for being far from mountain center (must be ON the mountain)
3. TIMEOUT per creature evaluation (prevents deadlocks)
4. Uses ProcessPoolExecutor with timeouts for robust multiprocessing
"""

import pybullet as p
import pybullet_data
import numpy as np
import os
import gc
from simulation import Simulation
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mountain peak location (center of arena, top of gaussian pyramid)
MOUNTAIN_PEAK = np.array([0, 0, 4])

# Timeout for individual creature evaluation (seconds)
CREATURE_TIMEOUT = 60


def calculate_stable_fitness(height_samples, final_pos, start_height=1.5):
    """
    Calculate fitness based on STABLE height achieved.

    This fitness function rewards creatures that:
    1. Climb high AND stay there (not just launch and fall)
    2. Move toward the mountain (progress bonus)
    3. Don't exploit physics by flying

    The key insight: we use the MINIMUM height in the final second of simulation.
    - Creatures that fly will eventually fall, giving low min height
    - Creatures that climb and stay will have consistent height

    Args:
        height_samples: List of height measurements from final portion of simulation
        final_pos: Final (x, y, z) position of creature
        start_height: Starting height to subtract

    Returns:
        float: Fitness score (higher = better climber)
    """
    if not height_samples:
        return 0.0

    # Use MINIMUM height from samples - this penalizes bouncing/flying
    # A creature that launches will have low min height when it falls
    # A creature stable on the mountain will have consistent height
    stable_height = min(height_samples)

    # Base fitness: stable height above starting position
    base_fitness = max(0, stable_height - start_height)

    # Calculate progress toward mountain center (x=0, y=0)
    # Creatures start at (0, -8, 1.5), so starting distance is 8
    start_dist = 8.0
    final_dist = np.sqrt(final_pos[0]**2 + final_pos[1]**2)

    # Progress bonus: reward moving TOWARD the mountain
    # Max bonus of 2.0 for reaching the center
    progress = start_dist - final_dist  # Positive if moved toward center
    progress_bonus = max(0, progress * 0.25)  # 0.25 per unit of progress

    # Small bonus for being on/near the mountain (within 3 units of center)
    if final_dist < 3:
        proximity_bonus = (3 - final_dist) * 0.3
    else:
        proximity_bonus = 0

    # Total fitness
    total_fitness = base_fitness + progress_bonus + proximity_bonus

    # Cap maximum fitness to prevent physics exploits
    # Mountain is ~4 units tall, realistic max climb is ~2.5 units
    # With bonuses, cap at 8
    capped_fitness = min(total_fitness, 8.0)

    return max(0, capped_fitness)


class MountainSimulation(Simulation):
    """
    Simulation class for mountain climbing creatures.
    Uses STABLE HEIGHT fitness - rewards climbing and staying, not flying.
    """

    def __init__(self, sim_id=0, landscape="gaussian_pyramid.urdf"):
        super().__init__(sim_id)
        self.landscape = landscape

    def run_creature(self, cr, iterations=2400):
        """
        Run a creature in the mountain environment and evaluate its climbing ability.

        Fitness is based on STABLE height - the minimum height achieved in the
        final second of simulation. This rewards creatures that climb and STAY
        on the mountain, not those that launch into the air.
        """
        pid = self.physicsClientId
        p.resetSimulation(physicsClientId=pid)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=pid)
        p.setGravity(0, 0, -10, physicsClientId=pid)

        # Build the Arena
        arena_size = 20
        wall_height = 1
        wall_thickness = 0.5

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

        # Load Mountain (configurable landscape)
        shapes_path = os.path.join(SCRIPT_DIR, 'shapes')
        p.setAdditionalSearchPath(shapes_path, physicsClientId=pid)
        mountain_pos = [0, 0, -1]
        mountain_orn = [0, 0, 0, 1]
        try:
            p.loadURDF(self.landscape, mountain_pos, mountain_orn, useFixedBase=1, physicsClientId=pid)
        except Exception as e:
            print(f"Error loading mountain: {e}. Ensure 'shapes/' directory exists with {self.landscape}")

        # Load Creature
        xml_file = os.path.join(SCRIPT_DIR, 'temp' + str(self.sim_id) + '.urdf')
        with open(xml_file, 'w') as f:
            f.write(cr.to_xml())

        # Spawn creature at edge of arena, facing the mountain
        start_pos = [0, -8, 1.5]
        cid = p.loadURDF(xml_file, start_pos, physicsClientId=pid)

        # Simulation Loop
        # Collect height samples from the FINAL portion of simulation
        # We use the last 240 steps (1 second at 240fps)
        final_sample_start = iterations - 240
        height_samples = []
        final_pos = start_pos

        for step in range(iterations):
            p.stepSimulation(physicsClientId=pid)

            if step % 24 == 0:
                self.update_motors(cid=cid, cr=cr)

            # Get position
            pos, _ = p.getBasePositionAndOrientation(cid, physicsClientId=pid)

            # Collect height samples from final second
            if step >= final_sample_start:
                height_samples.append(pos[2])

            # Track final position
            final_pos = pos
            cr.update_position(pos)

        # Calculate fitness using stable height metric
        cr.fitness = calculate_stable_fitness(height_samples, final_pos)


def _run_creature_worker(args):
    """
    Worker function for multiprocessing.
    Creates a fresh simulation in each worker process.
    Includes cleanup to prevent memory leaks.
    """
    sim_id, cr, iterations, landscape = args

    # Use process ID to avoid temp file conflicts
    unique_id = f"{sim_id}_{os.getpid()}"

    # Ensure motors are computed before running simulation
    cr.get_motors()

    sim = None
    try:
        sim = MountainSimulation(unique_id, landscape=landscape)
        sim.run_creature(cr, iterations)
    except Exception as e:
        # If simulation fails, assign zero fitness
        cr.fitness = 0
    finally:
        # Clean up PyBullet connection
        if sim is not None:
            try:
                p.disconnect(sim.physicsClientId)
            except:
                pass
        # Force garbage collection to prevent memory buildup
        gc.collect()

    return cr


class MountainThreadedSim:
    """
    Threaded simulation manager for parallel creature evaluation.
    Uses ProcessPoolExecutor with TIMEOUTS to prevent deadlocks.
    """

    def __init__(self, pool_size, landscape="gaussian_pyramid.urdf"):
        self.pool_size = pool_size
        self.landscape = landscape

    def eval_population(self, pop, iterations):
        """
        Evaluate all creatures in the population using multiprocessing.
        Uses timeouts to prevent individual creatures from blocking.
        """
        # Prepare arguments for worker function
        pool_args = []
        for i, cr in enumerate(pop.creatures):
            sim_id = i % self.pool_size
            pool_args.append((sim_id, cr, iterations, self.landscape))

        # Use ProcessPoolExecutor for better timeout support
        new_creatures = []

        # Process in batches to manage memory
        batch_size = self.pool_size * 2

        for batch_start in range(0, len(pool_args), batch_size):
            batch_end = min(batch_start + batch_size, len(pool_args))
            batch_args = pool_args[batch_start:batch_end]

            with ProcessPoolExecutor(max_workers=self.pool_size) as executor:
                # Submit all tasks
                futures = [executor.submit(_run_creature_worker, args) for args in batch_args]

                # Collect results with timeout
                for i, future in enumerate(futures):
                    try:
                        cr = future.result(timeout=CREATURE_TIMEOUT)
                        new_creatures.append(cr)
                    except FuturesTimeoutError:
                        # Creature evaluation timed out - assign zero fitness
                        cr = batch_args[i][1]  # Get original creature
                        cr.fitness = 0
                        new_creatures.append(cr)
                    except Exception as e:
                        # Other error - assign zero fitness
                        cr = batch_args[i][1]
                        cr.fitness = 0
                        new_creatures.append(cr)

            # Garbage collect between batches
            gc.collect()

        pop.creatures = new_creatures


class SensorMountainSimulation(Simulation):
    """
    Mountain simulation with sensory feedback.
    Creatures can sense the direction to the mountain peak and
    modulate their motor output accordingly.

    Uses STABLE HEIGHT fitness like MountainSimulation.
    """

    def __init__(self, sim_id=0, landscape="gaussian_pyramid.urdf"):
        super().__init__(sim_id)
        self.landscape = landscape

    def calculate_peak_alignment(self, cid, pid):
        """
        Calculate how well the creature is aligned with the mountain peak.

        Returns:
            float: Value from -1 to 1
                   1.0 = moving directly toward peak
                   -1.0 = moving directly away from peak
                   0.0 = stationary or perpendicular
        """
        # Get creature position and velocity
        pos, _ = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
        lin_vel, _ = p.getBaseVelocity(cid, physicsClientId=pid)

        # Vector from creature to mountain peak (2D, ignoring height)
        pos_2d = np.array([pos[0], pos[1]])
        peak_2d = np.array([MOUNTAIN_PEAK[0], MOUNTAIN_PEAK[1]])
        vec_to_peak = peak_2d - pos_2d

        # Normalize direction to peak
        dist_to_peak = np.linalg.norm(vec_to_peak)
        if dist_to_peak > 0.01:  # Avoid division by zero
            vec_to_peak = vec_to_peak / dist_to_peak
        else:
            return 0.0  # Already at peak

        # Get creature's movement direction
        heading = np.array([lin_vel[0], lin_vel[1]])
        speed = np.linalg.norm(heading)

        if speed > 0.01:  # Only if actually moving
            heading = heading / speed
            # Dot product gives alignment: 1 if same direction, -1 if opposite
            alignment = np.dot(vec_to_peak, heading)
            return float(alignment)

        return 0.0  # Not moving

    def update_motors_with_sensor(self, cid, cr, sensor_val):
        """
        Update motors with sensor feedback.
        """
        pid = self.physicsClientId
        for jid in range(p.getNumJoints(cid, physicsClientId=pid)):
            m = cr.get_motors()[jid]
            # Pass sensor value to motor - it will modulate output
            output = m.get_output(sensor_val)
            p.setJointMotorControl2(
                cid, jid,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=output,
                force=5,
                physicsClientId=pid
            )

    def run_creature(self, cr, iterations=2400):
        """
        Run a creature with sensory feedback enabled.
        Uses STABLE HEIGHT fitness.
        """
        pid = self.physicsClientId
        p.resetSimulation(physicsClientId=pid)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=pid)
        p.setGravity(0, 0, -10, physicsClientId=pid)

        # Build Arena
        arena_size = 20
        wall_height = 1
        wall_thickness = 0.5

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

        # Walls
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

        # Load Mountain (configurable landscape)
        shapes_path = os.path.join(SCRIPT_DIR, 'shapes')
        p.setAdditionalSearchPath(shapes_path, physicsClientId=pid)
        mountain_pos = [0, 0, -1]
        mountain_orn = [0, 0, 0, 1]
        try:
            p.loadURDF(self.landscape, mountain_pos, mountain_orn, useFixedBase=1, physicsClientId=pid)
        except Exception as e:
            print(f"Error loading mountain: {e}. Ensure 'shapes/' directory exists with {self.landscape}")

        # Load Creature
        xml_file = os.path.join(SCRIPT_DIR, 'temp' + str(self.sim_id) + '.urdf')
        with open(xml_file, 'w') as f:
            f.write(cr.to_xml())

        start_pos = [0, -8, 1.5]
        cid = p.loadURDF(xml_file, start_pos, physicsClientId=pid)

        # Simulation Loop with Sensor Feedback
        final_sample_start = iterations - 240
        height_samples = []
        final_pos = start_pos

        for step in range(iterations):
            p.stepSimulation(physicsClientId=pid)

            if step % 24 == 0:
                # Calculate sensor input (alignment with peak)
                sensor_val = self.calculate_peak_alignment(cid, pid)
                # Update motors with sensor feedback
                self.update_motors_with_sensor(cid, cr, sensor_val)

            # Get position
            pos, _ = p.getBasePositionAndOrientation(cid, physicsClientId=pid)

            # Collect height samples from final second
            if step >= final_sample_start:
                height_samples.append(pos[2])

            final_pos = pos
            cr.update_position(pos)

        # Calculate fitness using stable height metric
        cr.fitness = calculate_stable_fitness(height_samples, final_pos)


def _run_sensor_creature_worker(args):
    """
    Worker function for sensor-based multiprocessing.
    Creates a fresh simulation in each worker process.
    """
    sim_id, cr, iterations, landscape = args

    # Use process ID to avoid temp file conflicts
    unique_id = f"{sim_id}_{os.getpid()}"

    # Ensure motors are computed before running simulation
    cr.get_motors()

    sim = None
    try:
        sim = SensorMountainSimulation(unique_id, landscape=landscape)
        sim.run_creature(cr, iterations)
    except Exception as e:
        # If simulation fails, assign zero fitness
        cr.fitness = 0
    finally:
        # Clean up PyBullet connection
        if sim is not None:
            try:
                p.disconnect(sim.physicsClientId)
            except:
                pass
        # Force garbage collection
        gc.collect()

    return cr


class SensorMountainThreadedSim:
    """
    Threaded simulation with sensor feedback.
    Uses ProcessPoolExecutor with TIMEOUTS to prevent deadlocks.
    """

    def __init__(self, pool_size, landscape="gaussian_pyramid.urdf"):
        self.pool_size = pool_size
        self.landscape = landscape

    def eval_population(self, pop, iterations):
        """
        Evaluate all creatures with sensor feedback using multiprocessing.
        """
        # Prepare arguments for worker function
        pool_args = []
        for i, cr in enumerate(pop.creatures):
            sim_id = i % self.pool_size
            pool_args.append((sim_id, cr, iterations, self.landscape))

        # Use ProcessPoolExecutor for better timeout support
        new_creatures = []

        # Process in batches to manage memory
        batch_size = self.pool_size * 2

        for batch_start in range(0, len(pool_args), batch_size):
            batch_end = min(batch_start + batch_size, len(pool_args))
            batch_args = pool_args[batch_start:batch_end]

            with ProcessPoolExecutor(max_workers=self.pool_size) as executor:
                # Submit all tasks
                futures = [executor.submit(_run_sensor_creature_worker, args) for args in batch_args]

                # Collect results with timeout
                for i, future in enumerate(futures):
                    try:
                        cr = future.result(timeout=CREATURE_TIMEOUT)
                        new_creatures.append(cr)
                    except FuturesTimeoutError:
                        # Creature evaluation timed out - assign zero fitness
                        cr = batch_args[i][1]
                        cr.fitness = 0
                        new_creatures.append(cr)
                    except Exception as e:
                        # Other error - assign zero fitness
                        cr = batch_args[i][1]
                        cr.fitness = 0
                        new_creatures.append(cr)

            # Garbage collect between batches
            gc.collect()

        pop.creatures = new_creatures
