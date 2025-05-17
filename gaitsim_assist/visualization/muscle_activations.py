"""
Muscle activation visualization functions.

This module provides functions for plotting muscle activations from gait simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from ..simulation import SimulationResults


def plot_muscle_activations(results: SimulationResults,
                          muscles: Optional[List[str]] = None,
                          normalize_gait_cycle: bool = True,
                          show: bool = True,
                          save_path: Optional[Union[str, Path]] = None,
                          fig_size: tuple = (12, 8),
                          dpi: int = 100) -> plt.Figure:
    """Plot muscle activations from simulation results.
    
    Args:
        results: Simulation results
        muscles: List of muscles to plot. If None, all muscles are plotted.
        normalize_gait_cycle: Whether to normalize time to gait cycle percentage
        show: Whether to show the plot
        save_path: Path to save the plot
        fig_size: Figure size
        dpi: Figure DPI
        
    Returns:
        Matplotlib figure
    """
    # Get time and muscle activations
    time = results.time
    muscle_activations = results.muscle_activations
    
    # Select muscles to plot
    if muscles is None:
        muscles = list(muscle_activations.keys())
    
    # Limit the number of muscles to plot (to avoid overcrowding)
    max_muscles = 10
    if len(muscles) > max_muscles:
        print(f"Warning: Limiting plot to {max_muscles} muscles out of {len(muscles)}")
        muscles = muscles[:max_muscles]
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    
    # Normalize time to gait cycle percentage if requested
    if normalize_gait_cycle:
        x_values = np.linspace(0, 100, len(time))
        x_label = "Gait Cycle (%)"
    else:
        x_values = time
        x_label = "Time (s)"
    
    # Plot each muscle
    for muscle in muscles:
        if muscle in muscle_activations:
            ax.plot(x_values, muscle_activations[muscle], label=muscle)
    
    # Add labels and legend
    ax.set_xlabel(x_label)
    ax.set_ylabel("Activation")
    ax.set_title("Muscle Activations")
    ax.set_ylim(0, 1)  # Activations are between 0 and 1
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    
    return fig 