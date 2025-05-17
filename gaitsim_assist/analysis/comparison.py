"""
Comparison analysis module.

This module provides functions for comparing multiple simulation results
and analyzing the differences between them.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import matplotlib.pyplot as plt

from ..simulation import SimulationResults


def compare_simulations(results_list: List[SimulationResults],
                      labels: List[str],
                      metrics: Optional[List[str]] = None,
                      plot: bool = True,
                      save_path: Optional[str] = None) -> pd.DataFrame:
    """Compare multiple simulation results.
    
    Args:
        results_list: List of simulation results
        labels: List of labels for each simulation
        metrics: List of metrics to compare. If None, all available metrics are compared.
        plot: Whether to plot the comparison
        save_path: Path to save the plot
        
    Returns:
        DataFrame containing comparison results
    """
    if len(results_list) != len(labels):
        raise ValueError("Number of results must match number of labels")
    
    # Calculate metrics for each simulation
    all_metrics = []
    for results in results_list:
        from .metrics import calculate_gait_metrics
        metrics_dict = calculate_gait_metrics(results)
        all_metrics.append(metrics_dict)
    
    # Create a DataFrame with all metrics
    df = pd.DataFrame(all_metrics, index=labels)
    
    # Select specific metrics if provided
    if metrics is not None:
        df = df[metrics]
    
    # Plot the comparison if requested
    if plot:
        _plot_comparison(df, save_path)
    
    return df


def _plot_comparison(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """Plot a comparison of metrics.
    
    Args:
        df: DataFrame containing metrics to compare
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Determine the number of metrics to plot
    n_metrics = len(df.columns)
    
    # Limit the number of metrics to plot (to avoid overcrowding)
    max_metrics = 10
    if n_metrics > max_metrics:
        print(f"Warning: Limiting plot to {max_metrics} metrics out of {n_metrics}")
        df = df.iloc[:, :max_metrics]
        n_metrics = max_metrics
    
    # Create figure
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, n_metrics * 2), sharex=True)
    if n_metrics == 1:
        axes = [axes]  # Make sure axes is always a list
    
    # Plot each metric
    for i, metric in enumerate(df.columns):
        ax = axes[i]
        df[metric].plot(kind='bar', ax=ax)
        ax.set_title(metric)
        ax.set_ylabel('Value')
        ax.grid(True, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path is not None:
        plt.savefig(save_path)
    
    return fig


def calculate_differences(results1: SimulationResults,
                        results2: SimulationResults,
                        label1: str = "Simulation 1",
                        label2: str = "Simulation 2") -> pd.DataFrame:
    """Calculate differences between two simulation results.
    
    Args:
        results1: First simulation results
        results2: Second simulation results
        label1: Label for first simulation
        label2: Label for second simulation
        
    Returns:
        DataFrame containing differences
    """
    # Calculate metrics for each simulation
    from .metrics import calculate_gait_metrics
    metrics1 = calculate_gait_metrics(results1)
    metrics2 = calculate_gait_metrics(results2)
    
    # Compare metrics
    from .statistics import compare_metrics
    return compare_metrics(metrics1, metrics2, label1, label2) 