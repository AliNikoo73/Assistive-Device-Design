"""
Visualization module for GaitSim Assist.

This module provides tools for visualizing gait simulation results, including:
- Joint angle plots
- Ground reaction force plots
- Muscle activation plots
- Comparative visualizations between different simulations
"""

from .gait_plotter import GaitPlotter
from .joint_angles import plot_joint_angles
from .ground_forces import plot_ground_forces
from .muscle_activations import plot_muscle_activations
from .comparison import compare_simulations

__all__ = [
    'GaitPlotter',
    'plot_joint_angles',
    'plot_ground_forces',
    'plot_muscle_activations',
    'compare_simulations'
] 