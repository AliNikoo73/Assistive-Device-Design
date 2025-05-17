"""
Joint angle visualization functions.

This module provides functions for plotting joint angles from gait simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from ..simulation import SimulationResults


def plot_joint_angles(results: SimulationResults,
                     joints: Optional[List[str]] = None,
                     normalize_gait_cycle: bool = True,
                     show: bool = True,
                     save_path: Optional[Union[str, Path]] = None,
                     fig_size: tuple = (10, 6),
                     dpi: int = 100) -> plt.Figure:
    """Plot joint angles from simulation results.
    
    Args:
        results: Simulation results
        joints: List of joints to plot. If None, all joints are plotted.
        normalize_gait_cycle: Whether to normalize time to gait cycle percentage
        show: Whether to show the plot
        save_path: Path to save the plot
        fig_size: Figure size
        dpi: Figure DPI
        
    Returns:
        Matplotlib figure
    """
    # Get time and joint angles
    time = results.time
    joint_angles = results.joint_angles
    
    # Select joints to plot
    if joints is None:
        joints = list(joint_angles.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    
    # Normalize time to gait cycle percentage if requested
    if normalize_gait_cycle:
        x_values = np.linspace(0, 100, len(time))
        x_label = "Gait Cycle (%)"
    else:
        x_values = time
        x_label = "Time (s)"
    
    # Plot each joint
    for joint in joints:
        if joint in joint_angles:
            ax.plot(x_values, joint_angles[joint], label=joint)
    
    # Add labels and legend
    ax.set_xlabel(x_label)
    ax.set_ylabel("Joint Angle (deg)")
    ax.set_title("Joint Angles")
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