"""
Ground reaction force visualization functions.

This module provides functions for plotting ground reaction forces from gait simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from ..simulation import SimulationResults


def plot_ground_forces(results: SimulationResults,
                      force_components: Optional[List[str]] = None,
                      normalize_gait_cycle: bool = True,
                      show: bool = True,
                      save_path: Optional[Union[str, Path]] = None,
                      fig_size: tuple = (10, 6),
                      dpi: int = 100) -> plt.Figure:
    """Plot ground reaction forces from simulation results.
    
    Args:
        results: Simulation results
        force_components: List of force components to plot. If None, all components are plotted.
        normalize_gait_cycle: Whether to normalize time to gait cycle percentage
        show: Whether to show the plot
        save_path: Path to save the plot
        fig_size: Figure size
        dpi: Figure DPI
        
    Returns:
        Matplotlib figure
    """
    # Get time and ground forces
    time = results.time
    ground_forces = results.ground_forces
    
    # Select force components to plot
    if force_components is None:
        force_components = list(ground_forces.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    
    # Normalize time to gait cycle percentage if requested
    if normalize_gait_cycle:
        x_values = np.linspace(0, 100, len(time))
        x_label = "Gait Cycle (%)"
    else:
        x_values = time
        x_label = "Time (s)"
    
    # Plot each force component
    for component in force_components:
        if component in ground_forces:
            ax.plot(x_values, ground_forces[component], label=component)
    
    # Add labels and legend
    ax.set_xlabel(x_label)
    ax.set_ylabel("Force (N)")
    ax.set_title("Ground Reaction Forces")
    ax.legend()
    ax.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    # Show if requested
    if show:
        plt.show()
    
    return fig 