#!/usr/bin/env python3
"""
Example of optimizing assistive device parameters with GaitSim Assist.

This example demonstrates how to:
1. Create an assistive device (exoskeleton)
2. Optimize its parameters to minimize metabolic cost
3. Visualize the results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Dict, Any

import gaitsim_assist as gsa
from gaitsim_assist.devices import Exoskeleton
from gaitsim_assist.visualization import GaitPlotter


def evaluate_exoskeleton(simulator: gsa.GaitSimulator, 
                        max_torque: float, 
                        mass: float) -> Dict[str, Any]:
    """Evaluate an exoskeleton with the given parameters.
    
    Args:
        simulator: GaitSimulator instance
        max_torque: Maximum torque of the exoskeleton (Nm)
        mass: Mass of the exoskeleton (kg)
        
    Returns:
        Dictionary with evaluation results
    """
    # Create an ankle exoskeleton with the given parameters
    exo = gsa.devices.Exoskeleton(
        name="ankle_exo",
        model=simulator.model,
        joint_name="ankle",
        mass=mass,
        max_torque=max_torque
    )
    
    # Run a predictive simulation with the exoskeleton
    results = simulator.run_predictive_simulation(
        cost_function='cot',
        time_range=(0.0, 1.0),
        assistive_device=exo
    )
    
    # Return the metabolic cost and other metrics
    return {
        'max_torque': max_torque,
        'mass': mass,
        'metabolic_cost': results.metabolic_cost,
        'results': results
    }


def main():
    # Create output directory
    output_dir = Path("simulation_results")
    output_dir.mkdir(exist_ok=True)
    
    print("Creating gait simulator...")
    # Create a gait simulator with default 2D walking model
    simulator = gsa.GaitSimulator()
    
    # Run a baseline simulation without an assistive device
    print("Running baseline simulation...")
    baseline_results = simulator.run_predictive_simulation(
        cost_function='cot',
        time_range=(0.0, 1.0)
    )
    
    # Define the parameter space to explore
    max_torques = [25.0, 50.0, 75.0]  # Nm
    masses = [0.5, 1.0, 1.5]  # kg
    
    # Initialize results storage
    all_results = []
    best_result = None
    best_metabolic_cost = float('inf')
    
    # Evaluate each parameter combination
    print("Optimizing exoskeleton parameters...")
    for max_torque in max_torques:
        for mass in masses:
            print(f"  Evaluating max_torque={max_torque} Nm, mass={mass} kg...")
            result = evaluate_exoskeleton(simulator, max_torque, mass)
            all_results.append(result)
            
            # Check if this is the best result so far
            if result['metabolic_cost'] < best_metabolic_cost:
                best_metabolic_cost = result['metabolic_cost']
                best_result = result
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame(all_results)
    
    # Save the results to a CSV file
    results_df.to_csv(output_dir / "exoskeleton_optimization_results.csv", index=False)
    
    # Print the best parameters
    print("\nOptimization results:")
    print(f"  Best max_torque: {best_result['max_torque']} Nm")
    print(f"  Best mass: {best_result['mass']} kg")
    print(f"  Metabolic cost: {best_result['metabolic_cost']:.2f} J/kg/m")
    print(f"  Improvement over baseline: {(baseline_results.metabolic_cost - best_result['metabolic_cost']) / baseline_results.metabolic_cost * 100:.2f}%")
    
    # Visualize the results
    print("\nVisualizing results...")
    plotter = GaitPlotter()
    
    # Compare joint angles between baseline and best exoskeleton
    plotter.compare_simulations(
        results_list=[baseline_results, best_result['results']],
        labels=["Baseline", "Optimized Exoskeleton"],
        plot_type="joint_angles",
        normalize_gait_cycle=True,
        save_path=output_dir / "exo_comparison_joint_angles.png"
    )
    
    # Compare ground reaction forces
    plotter.compare_simulations(
        results_list=[baseline_results, best_result['results']],
        labels=["Baseline", "Optimized Exoskeleton"],
        plot_type="ground_forces",
        normalize_gait_cycle=True,
        save_path=output_dir / "exo_comparison_ground_forces.png"
    )
    
    # Plot metabolic cost as a function of parameters
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a pivot table for the heatmap
    pivot_table = results_df.pivot(index='mass', columns='max_torque', values='metabolic_cost')
    
    # Create a heatmap
    im = ax.imshow(pivot_table.values, cmap='viridis_r')
    
    # Set labels
    ax.set_xticks(np.arange(len(max_torques)))
    ax.set_yticks(np.arange(len(masses)))
    ax.set_xticklabels(max_torques)
    ax.set_yticklabels(masses)
    ax.set_xlabel('Maximum Torque (Nm)')
    ax.set_ylabel('Mass (kg)')
    ax.set_title('Metabolic Cost (J/kg/m)')
    
    # Add colorbar
    cbar = fig.colorbar(im)
    
    # Add text annotations
    for i in range(len(masses)):
        for j in range(len(max_torques)):
            text = ax.text(j, i, f"{pivot_table.values[i, j]:.2f}",
                         ha="center", va="center", color="white")
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_dir / "exo_parameter_heatmap.png", dpi=300)
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main() 