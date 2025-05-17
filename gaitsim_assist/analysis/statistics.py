"""
Statistical analysis module for gait data.

This module provides functions for performing statistical analysis on gait data,
such as comparing metrics between different simulations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
from scipy import stats

from ..simulation import SimulationResults


def run_statistical_analysis(results_list: List[SimulationResults],
                           labels: List[str],
                           metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """Run statistical analysis on multiple simulation results.
    
    Args:
        results_list: List of simulation results
        labels: List of labels for each simulation
        metrics: List of metrics to analyze. If None, all available metrics are analyzed.
        
    Returns:
        DataFrame containing statistical analysis results
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
    
    # Calculate basic statistics
    stats_df = pd.DataFrame({
        'mean': df.mean(),
        'std': df.std(),
        'min': df.min(),
        'max': df.max()
    })
    
    # If we have enough samples, calculate p-values
    if len(results_list) >= 2:
        # Perform t-tests between first simulation and others
        p_values = {}
        for i in range(1, len(labels)):
            p_vals = {}
            for col in df.columns:
                # Skip if there are NaNs
                if np.isnan(df.iloc[0][col]) or np.isnan(df.iloc[i][col]):
                    p_vals[col] = np.nan
                else:
                    # Simple t-test (in a real implementation, we would need more samples)
                    t_stat, p_val = stats.ttest_ind(
                        [df.iloc[0][col]], [df.iloc[i][col]], 
                        equal_var=False
                    )
                    p_vals[col] = p_val
            p_values[f'p_value_{labels[0]}_vs_{labels[i]}'] = p_vals
        
        # Add p-values to stats DataFrame
        for name, p_vals in p_values.items():
            stats_df[name] = pd.Series(p_vals)
    
    return stats_df


def compare_metrics(metrics1: Dict[str, float], 
                  metrics2: Dict[str, float],
                  label1: str = "Simulation 1",
                  label2: str = "Simulation 2") -> pd.DataFrame:
    """Compare metrics between two simulations.
    
    Args:
        metrics1: Metrics from first simulation
        metrics2: Metrics from second simulation
        label1: Label for first simulation
        label2: Label for second simulation
        
    Returns:
        DataFrame comparing the metrics
    """
    # Create a DataFrame with both sets of metrics
    df = pd.DataFrame({
        label1: pd.Series(metrics1),
        label2: pd.Series(metrics2)
    })
    
    # Calculate absolute and percentage differences
    df['abs_diff'] = df[label2] - df[label1]
    df['pct_diff'] = (df[label2] - df[label1]) / df[label1] * 100
    
    return df 