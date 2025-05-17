"""
Analysis module for GaitSim Assist.

This module provides tools for analyzing gait simulation results, including:
- Gait metrics calculation
- Statistical analysis
- Comparison between different simulations
"""

from .metrics import calculate_gait_metrics
from .statistics import run_statistical_analysis
from .comparison import compare_simulations

__all__ = [
    'calculate_gait_metrics',
    'run_statistical_analysis',
    'compare_simulations'
] 