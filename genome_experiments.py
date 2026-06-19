"""
Genome Experiments Module

This module provides modified genome specifications for different
evolutionary experiments as described in the coursework brief.

Experiment A: Snake-Like Morphology
- Reduced link-recurrence (prevents branching)
- Thinner limbs (reduced link-radius)
- Shorter segments (reduced link-length)

Experiment B: Control-Only Evolution
- Fixed morphology genes
- Only control parameters (waveform, amp, freq) are mutated
"""

import numpy as np
import copy
import random
from genome import Genome


class SnakeGenome(Genome):
    """
    Genome variant that produces snake-like creatures.
    Constrains morphology to long, thin, chain-like bodies.
    """

    @staticmethod
    def get_gene_spec():
        """
        Modified gene spec for snake-like morphology.
        Key changes:
        - link-recurrence: scale=1 (was 3) - prevents branching
        - link-radius: scale=0.1 (was 1) - thinner limbs
        - link-length: scale=1.0 (was 2) - moderate length segments
        """
        gene_spec = {
            "link-shape": {"scale": 1},
            "link-length": {"scale": 1.0},      # Reduced from 2
            "link-radius": {"scale": 0.1},      # Reduced from 1 (thinner)
            "link-recurrence": {"scale": 1},    # Reduced from 3 (no branching)
            "link-mass": {"scale": 0.5},        # Lighter
            "joint-type": {"scale": 1},
            "joint-parent": {"scale": 1},
            "joint-axis-xyz": {"scale": 1},
            "joint-origin-rpy-1": {"scale": np.pi * 2},
            "joint-origin-rpy-2": {"scale": np.pi * 2},
            "joint-origin-rpy-3": {"scale": np.pi * 2},
            "joint-origin-xyz-1": {"scale": 1},
            "joint-origin-xyz-2": {"scale": 1},
            "joint-origin-xyz-3": {"scale": 1},
            "control-waveform": {"scale": 1},
            "control-amp": {"scale": 0.25},
            "control-freq": {"scale": 1}
        }
        ind = 0
        for key in gene_spec.keys():
            gene_spec[key]["ind"] = ind
            ind = ind + 1
        return gene_spec


class HeavyGenome(Genome):
    """
    Genome variant that produces heavier, blockier creatures.
    Good for stability on slopes.
    """

    @staticmethod
    def get_gene_spec():
        """
        Modified gene spec for heavy/blocky morphology.
        - Larger radius for more stable base
        - Higher mass
        - Limited recurrence
        """
        gene_spec = {
            "link-shape": {"scale": 1},
            "link-length": {"scale": 1.5},
            "link-radius": {"scale": 1.5},      # Increased (thicker)
            "link-recurrence": {"scale": 2},    # Moderate branching
            "link-mass": {"scale": 2.0},        # Heavier
            "joint-type": {"scale": 1},
            "joint-parent": {"scale": 1},
            "joint-axis-xyz": {"scale": 1},
            "joint-origin-rpy-1": {"scale": np.pi * 2},
            "joint-origin-rpy-2": {"scale": np.pi * 2},
            "joint-origin-rpy-3": {"scale": np.pi * 2},
            "joint-origin-xyz-1": {"scale": 1},
            "joint-origin-xyz-2": {"scale": 1},
            "joint-origin-xyz-3": {"scale": 1},
            "control-waveform": {"scale": 1},
            "control-amp": {"scale": 0.3},      # Slightly stronger motors
            "control-freq": {"scale": 1}
        }
        ind = 0
        for key in gene_spec.keys():
            gene_spec[key]["ind"] = ind
            ind = ind + 1
        return gene_spec


class ControlOnlyGenome(Genome):
    """
    Genome variant where only control parameters are evolved.
    Morphology genes are fixed at initialization.
    """

    # Class-level fixed morphology template
    _fixed_morphology = None

    @staticmethod
    def set_fixed_morphology(dna):
        """Set the fixed morphology template from existing DNA."""
        ControlOnlyGenome._fixed_morphology = copy.deepcopy(dna)

    @staticmethod
    def get_gene_spec():
        """Standard gene spec - same as base Genome."""
        return Genome.get_gene_spec()

    @staticmethod
    def point_mutate(genome, rate, amount):
        """
        Only mutate control genes (indices 14, 15, 16).
        Morphology genes (0-13) remain unchanged.
        """
        spec = Genome.get_gene_spec()
        control_indices = [
            spec["control-waveform"]["ind"],
            spec["control-amp"]["ind"],
            spec["control-freq"]["ind"]
        ]

        new_genome = copy.copy(genome)
        for gene in new_genome:
            for i in control_indices:  # Only mutate control genes
                if random.random() < rate:
                    gene[i] += random.uniform(-amount, amount)
                if gene[i] >= 1.0:
                    gene[i] = 0.9999
                if gene[i] < 0.0:
                    gene[i] = 0.0
        return new_genome

    @staticmethod
    def shrink_mutate(genome, rate):
        """Disabled - don't change number of genes in control-only mode."""
        return copy.copy(genome)

    @staticmethod
    def grow_mutate(genome, rate):
        """Disabled - don't change number of genes in control-only mode."""
        return copy.copy(genome)

    @staticmethod
    def crossover(g1, g2):
        """
        Crossover only control genes.
        Morphology from parent 1 is preserved.
        """
        spec = Genome.get_gene_spec()
        control_indices = [
            spec["control-waveform"]["ind"],
            spec["control-amp"]["ind"],
            spec["control-freq"]["ind"]
        ]

        # Start with copy of g1 (preserves morphology)
        new_genome = copy.deepcopy(g1)

        # For each gene, randomly pick control values from either parent
        min_len = min(len(g1), len(g2))
        for i in range(min_len):
            if random.random() < 0.5:
                # Take control genes from g2
                for idx in control_indices:
                    new_genome[i][idx] = g2[i][idx]

        return new_genome


class LeggedGenome(Genome):
    """
    Genome variant optimized for legged locomotion.
    Encourages symmetric, multi-legged body plans.
    """

    @staticmethod
    def get_gene_spec():
        """
        Gene spec encouraging leg-like appendages.
        - Higher recurrence for multiple legs
        - Moderate length/radius for leg segments
        """
        gene_spec = {
            "link-shape": {"scale": 1},
            "link-length": {"scale": 1.2},
            "link-radius": {"scale": 0.3},
            "link-recurrence": {"scale": 4},    # More recurrence for legs
            "link-mass": {"scale": 0.8},
            "joint-type": {"scale": 1},
            "joint-parent": {"scale": 0.5},     # Prefer attaching to root
            "joint-axis-xyz": {"scale": 1},
            "joint-origin-rpy-1": {"scale": np.pi * 2},
            "joint-origin-rpy-2": {"scale": np.pi * 2},
            "joint-origin-rpy-3": {"scale": np.pi * 2},
            "joint-origin-xyz-1": {"scale": 1},
            "joint-origin-xyz-2": {"scale": 1},
            "joint-origin-xyz-3": {"scale": 1},
            "control-waveform": {"scale": 1},
            "control-amp": {"scale": 0.25},
            "control-freq": {"scale": 1.5}      # Faster movement
        }
        ind = 0
        for key in gene_spec.keys():
            gene_spec[key]["ind"] = ind
            ind = ind + 1
        return gene_spec


# Convenience function to get genome class by name
def get_genome_class(name):
    """
    Get a genome class by experiment name.

    Args:
        name: One of 'standard', 'snake', 'heavy', 'control_only', 'legged'

    Returns:
        The corresponding Genome class
    """
    genome_classes = {
        'standard': Genome,
        'snake': SnakeGenome,
        'heavy': HeavyGenome,
        'control_only': ControlOnlyGenome,
        'legged': LeggedGenome
    }
    return genome_classes.get(name, Genome)
