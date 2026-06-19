"""
Simulation module for PyBullet physics evaluation.
Provides single-threaded and multi-threaded simulation runners
for evaluating creature fitness in the physics environment.
"""
import pybullet as p
from multiprocessing import Pool


class Simulation:
    """
    Single-threaded PyBullet simulation for creature evaluation.
    Runs in DIRECT mode (headless) for fast fitness evaluation.
    """

    def __init__(self, sim_id=0):
        """Initialise PyBullet physics client in headless mode."""
        self.physicsClientId = p.connect(p.DIRECT)
        self.sim_id = sim_id

    def run_creature(self, cr, iterations=2400):
        """
        Evaluate a creature in the physics simulation.

        Args:
            cr: Creature object to evaluate
            iterations: Simulation steps (2400 = 10 seconds at 240Hz)
        """
        pid = self.physicsClientId
        p.resetSimulation(physicsClientId=pid)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)

        # Set up environment with gravity and ground plane
        p.setGravity(0, 0, -10, physicsClientId=pid)
        plane_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=pid)
        floor = p.createMultiBody(plane_shape, plane_shape, physicsClientId=pid)

        # Export creature to URDF and load into simulation
        xml_file = 'temp' + str(self.sim_id) + '.urdf'
        xml_str = cr.to_xml()
        with open(xml_file, 'w') as f:
            f.write(xml_str)

        cid = p.loadURDF(xml_file, physicsClientId=pid)
        p.resetBasePositionAndOrientation(cid, [0, 0, 2.5], [0, 0, 0, 1], physicsClientId=pid)

        # Run simulation loop
        for step in range(iterations):
            p.stepSimulation(physicsClientId=pid)
            # Update motor outputs every 24 steps (10Hz control rate)
            if step % 24 == 0:
                self.update_motors(cid=cid, cr=cr)

            pos, orn = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
            cr.update_position(pos)

    def update_motors(self, cid, cr):
        """
        Apply motor outputs to creature joints.

        Args:
            cid: PyBullet body ID for the creature
            cr: Creature object containing motor controllers
        """
        for jid in range(p.getNumJoints(cid, physicsClientId=self.physicsClientId)):
            m = cr.get_motors()[jid]
            p.setJointMotorControl2(cid, jid,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=m.get_output(),
                                    force=5,
                                    physicsClientId=self.physicsClientId)

    def eval_population(self, pop, iterations):
        """Evaluate all creatures in population sequentially."""
        for cr in pop.creatures:
            self.run_creature(cr, 2400)


class ThreadedSim:
    """
    Multi-threaded simulation using process pool for parallel evaluation.
    Creates multiple simulation instances for concurrent creature testing.
    """

    def __init__(self, pool_size):
        """Create pool of simulation instances for parallel execution."""
        self.sims = [Simulation(i) for i in range(pool_size)]

    @staticmethod
    def static_run_creature(sim, cr, iterations):
        """Static method for multiprocessing - runs creature and returns result."""
        sim.run_creature(cr, iterations)
        return cr

    def eval_population(self, pop, iterations):
        """
        Evaluate population in parallel using process pool.

        Args:
            pop: Population object containing creatures
            iterations: Simulation steps per creature (2400 = 10 sec at 240Hz)
        """
        pool_args = []
        start_ind = 0
        pool_size = len(self.sims)

        # Batch creatures into groups matching pool size
        while start_ind < len(pop.creatures):
            this_pool_args = []
            for i in range(start_ind, start_ind + pool_size):
                if i == len(pop.creatures):
                    break
                sim_ind = i % len(self.sims)
                this_pool_args.append([
                    self.sims[sim_ind],
                    pop.creatures[i],
                    iterations]
                )
            pool_args.append(this_pool_args)
            start_ind = start_ind + pool_size

        # Execute batches and collect results
        new_creatures = []
        for pool_argset in pool_args:
            with Pool(pool_size) as p:
                creatures = p.starmap(ThreadedSim.static_run_creature, pool_argset)
                new_creatures.extend(creatures)
        pop.creatures = new_creatures
