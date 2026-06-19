"""
Creature module for evolutionary robotics.
Defines the phenotype representation of evolved creatures including
their physical structure (links/joints) and motor control systems.
"""
import genome
from xml.dom.minidom import getDOMImplementation
from enum import Enum
import numpy as np


class MotorType(Enum):
    """Enumeration of available motor control waveform types."""
    PULSE = 1
    SINE = 2

class Motor:
    """
    Motor controller for creature joints.
    Generates oscillating output signals (pulse or sine wave) that can be
    modulated by sensory input for reactive behaviour.
    """

    def __init__(self, control_waveform, control_amp, control_freq):
        """Initialise motor with waveform type, amplitude and frequency."""
        if control_waveform <= 0.5:
            self.motor_type = MotorType.PULSE
        else:
            self.motor_type = MotorType.SINE
        self.amp = control_amp
        self.freq = control_freq
        self.phase = 0
        # Sensor gain: how strongly the motor responds to sensor input
        # Higher values = more reactive to mountain direction
        self.sensor_gain = 2.0

    def get_output(self, sensor_input=0):
        """
        Calculate motor output with optional sensor modulation.

        Args:
            sensor_input: Value from -1 to 1 indicating alignment with target.
                         1.0 = facing mountain peak
                         -1.0 = facing away from peak
                         0.0 = perpendicular or no sensor data

        Returns:
            Motor velocity output, modulated by sensor input if provided.
        """
        self.phase = (self.phase + self.freq) % (np.pi * 2)
        if self.motor_type == MotorType.PULSE:
            if self.phase < np.pi:
                output = 1
            else:
                output = -1

        if self.motor_type == MotorType.SINE:
            output = np.sin(self.phase)

        # Base output scaled by amplitude
        base_output = output * self.amp

        # Sensor modulation: boost when facing peak, reduce when facing away
        # This encourages creatures to move toward the mountain
        modulated_output = base_output + (sensor_input * self.sensor_gain * self.amp)

        return modulated_output 

class Creature:
    """
    Represents an evolved creature with genome, morphology, and motors.
    The creature's physical structure is generated from its DNA through
    a genotype-to-phenotype mapping process.
    """

    def __init__(self, gene_count):
        """Create creature with random genome of specified gene count."""
        self.spec = genome.Genome.get_gene_spec()
        self.dna = genome.Genome.get_random_genome(len(self.spec), gene_count)
        self.flat_links = None   # Unexpanded link structure
        self.exp_links = None    # Fully expanded links for URDF
        self.motors = None       # Motor controllers for each joint
        self.start_position = None
        self.last_position = None

    def get_flat_links(self):
        """Convert genome to flat link structure (lazy evaluation)."""
        if self.flat_links == None:
            gdicts = genome.Genome.get_genome_dicts(self.dna, self.spec)
            self.flat_links = genome.Genome.genome_to_links(gdicts)
        return self.flat_links
    
    def get_expanded_links(self):
        """Expand recursive link structure into flat list for URDF generation."""
        self.get_flat_links()
        if self.exp_links is not None:
            return self.exp_links

        exp_links = [self.flat_links[0]]
        genome.Genome.expandLinks(self.flat_links[0],
                                  self.flat_links[0].name,
                                  self.flat_links,
                                  exp_links)
        self.exp_links = exp_links
        return self.exp_links

    def to_xml(self):
        """Generate URDF XML representation for PyBullet simulation."""
        self.get_expanded_links()
        domimpl = getDOMImplementation()
        adom = domimpl.createDocument(None, "start", None)
        robot_tag = adom.createElement("robot")
        for link in self.exp_links:
            robot_tag.appendChild(link.to_link_element(adom))
        first = True
        for link in self.exp_links:
            if first:# skip the root node! 
                first = False
                continue
            robot_tag.appendChild(link.to_joint_element(adom))
        robot_tag.setAttribute("name", "pepe") #  choose a name!
        return '<?xml version="1.0"?>' + robot_tag.toprettyxml()

    def get_motors(self):
        """Create motor controllers for each joint from link parameters."""
        self.get_expanded_links()
        if self.motors == None:
            motors = []
            for i in range(1, len(self.exp_links)):
                l = self.exp_links[i]
                m = Motor(l.control_waveform, l.control_amp, l.control_freq)
                motors.append(m)
            self.motors = motors
        return self.motors

    def update_position(self, pos):
        """Track creature position for distance calculations."""
        if self.start_position == None:
            self.start_position = pos
        else:
            self.last_position = pos

    def get_distance_travelled(self):
        """Calculate Euclidean distance from start to current position."""
        if self.start_position is None or self.last_position is None:
            return 0
        p1 = np.asarray(self.start_position)
        p2 = np.asarray(self.last_position)
        dist = np.linalg.norm(p1 - p2)
        return dist

    def update_dna(self, dna):
        """Replace genome and reset derived attributes for re-evaluation."""
        self.dna = dna
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None