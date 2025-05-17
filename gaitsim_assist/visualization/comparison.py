"""
Comparison visualization functions.

This module provides functions for comparing multiple simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from ..simulation import SimulationResults


def compare_simulations(results_list: List[SimulationResults],
                      labels: List[str],
                      plot_type: str = "joint_angles",
                      items: Optional[List[str]] = None,
                      normalize_gait_cycle: bool = True,
                      show: bool = True,
                      save_path: Optional[Union[str, Path]] = None,
                      fig_size: tuple = (12, 8),
                      dpi: int = 100) -> plt.Figure:
    """Compare multiple simulation results.
    
    Args:
        results_list: List of simulation results
        labels: List of labels for each simulation
        plot_type: Type of data to plot ('joint_angles', 'ground_forces', or 'muscle_activations')
        items: List of specific items to plot. If None, common items across all simulations are plotted.
        normalize_gait_cycle: Whether to normalize time to gait cycle percentage
        show: Whether to show the plot
        save_path: Path to save the plot
        fig_size: Figure size
        dpi: Figure DPI
        
    Returns:
        Matplotlib figure
    """
    if len(results_list) != len(labels):
        raise ValueError("Number of results must match number of labels")
    
    # Determine what data to plot
    if plot_type == "joint_angles":
        data_attr = "joint_angles"
        y_label = "Joint Angle (deg)"
        title = "Joint Angles Comparison"
    elif plot_type == "ground_forces":
        data_attr = "ground_forces"
        y_label = "Force (N)"
        title = "Ground Reaction Forces Comparison"
    elif plot_type == "muscle_activations":
        data_attr = "muscle_activations"
        y_label = "Activation"
        title = "Muscle Activations Comparison"
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
    
    # Get common items across all simulations if not specified
    if items is None:
        common_items = set(getattr(results_list[0], data_attr).keys())
        for results in results_list[1:]:
            common_items &= set(getattr(results, data_attr).keys())
        items = list(common_items)
    
    # Limit the number of items to plot (to avoid overcrowding)
    max_items = 4
    if len(items) > max_items:
        print(f"Warning: Limiting plot to {max_items} items out of {len(items)}")
        items = items[:max_items]
    
    # Create figure with subplots for each item
    fig, axes = plt.subplots(len(items), 1, figsize=fig_size, dpi=dpi, sharex=True)
    if len(items) == 1:
        axes = [axes]  # Make sure axes is always a list
    
    # Plot each item
    for i, item in enumerate(items):
        ax = axes[i]
        
        for j, (results, label) in enumerate(zip(results_list, labels)):
            # Get data for this item
            data = getattr(results, data_attr).get(item)
            if data is None:
                continue
            
            # Get time and normalize if requested
            time = results.time
            if normalize_gait_cycle:
                x_values = np.linspace(0, 100, len(time))
                x_label = "Gait Cycle (%)"
            else:
                x_values = time
                x_label = "Time (s)"
            
            # Plot data
            ax.plot(x_values, data, label=label)
        
        # Add labels and legend
        ax.set_ylabel(y_label)
        ax.set_title(item)
        ax.grid(True)
        
        # Only add legend to the first subplot
        if i == 0:
            ax.legend()
    
    # Add x-label to the bottom subplot
    axes[-1].set_xlabel(x_label)
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Save if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    # Show if requested
    if show:
        plt.show()
    
    return fig 