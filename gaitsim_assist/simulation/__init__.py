"""
Simulation module for GaitSim Assist.

This module provides the core simulation capabilities, including:
- GaitSimulator: Main class for running gait simulations
- TrackingSimulation: For tracking experimental gait data
- PredictiveSimulation: For predictive simulations with various cost functions
"""

from .gait_simulator import GaitSimulator, SimulationResults
from .tracking import TrackingSimulation
from .predictive import PredictiveSimulation

__all__ = [
    'GaitSimulator',
    'SimulationResults',
    'TrackingSimulation',
    'PredictiveSimulation'
] 