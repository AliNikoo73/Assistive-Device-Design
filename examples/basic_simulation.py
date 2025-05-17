#!/usr/bin/env python3
"""
Basic example of using GaitSim Assist to run a gait simulation.

This example demonstrates how to:
1. Create a GaitSimulator instance
2. Run a predictive simulation with different cost functions
3. Visualize the results
4. Compare simulations with different cost functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import gaitsim_assist as gsa
from gaitsim_assist.visualization import GaitPlotter


def main():
    # Create output directory
    output_dir = Path("simulation_results")
    output_dir.mkdir(exist_ok=True)
    
    print("Creating gait simulator...")
    # Create a gait simulator with default 2D walking model
    simulator = gsa.GaitSimulator()
    
    # Run a predictive simulation with cost of transport cost function
    print("Running simulation with cost of transport...")
    cot_results = simulator.run_predictive_simulation(
        cost_function='cot',
        time_range=(0.0, 1.0)
    )
    
    # Run a predictive simulation with muscle effort cost function
    print("Running simulation with muscle effort...")
    effort_results = simulator.run_predictive_simulation(
        cost_function='muscle_effort',
        time_range=(0.0, 1.0)
    )
    
    # Create a hybrid cost function (50% CoT, 50% muscle effort)
    print("Running simulation with hybrid cost function...")
    hybrid_cost = gsa.cost_functions.Hybrid(
        simulator.model,
        cost_functions={
            'cot': 0.5,
            'muscle_effort': 0.5
        }
    )
    hybrid_results = simulator.run_predictive_simulation(
        cost_function=hybrid_cost,
        time_range=(0.0, 1.0)
    )
    
    # Create an ankle exoskeleton
    print("Running simulation with ankle exoskeleton...")
    exo = gsa.devices.Exoskeleton(
        name="ankle_exo",
        model=simulator.model,
        joint_name="ankle",
        mass=1.0,
        max_torque=50.0
    )
    exo_results = simulator.run_predictive_simulation(
        cost_function='cot',
        time_range=(0.0, 1.0),
        assistive_device=exo
    )
    
    # Visualize the results
    print("Visualizing results...")
    plotter = GaitPlotter()
    
    # Plot joint angles for cost of transport simulation
    plotter.plot_joint_angles(
        cot_results,
        normalize_gait_cycle=True,
        save_path=output_dir / "cot_joint_angles.png"
    )
    
    # Plot ground reaction forces for cost of transport simulation
    plotter.plot_ground_forces(
        cot_results,
        normalize_gait_cycle=True,
        save_path=output_dir / "cot_ground_forces.png"
    )
    
    # Plot muscle activations for cost of transport simulation
    plotter.plot_muscle_activations(
        cot_results,
        normalize_gait_cycle=True,
        save_path=output_dir / "cot_muscle_activations.png"
    )
    
    # Compare joint angles between different simulations
    plotter.compare_simulations(
        results_list=[cot_results, effort_results, hybrid_results, exo_results],
        labels=["Cost of Transport", "Muscle Effort", "Hybrid", "Exoskeleton"],
        plot_type="joint_angles",
        normalize_gait_cycle=True,
        save_path=output_dir / "comparison_joint_angles.png"
    )
    
    # Compare ground reaction forces between different simulations
    plotter.compare_simulations(
        results_list=[cot_results, effort_results, hybrid_results, exo_results],
        labels=["Cost of Transport", "Muscle Effort", "Hybrid", "Exoskeleton"],
        plot_type="ground_forces",
        normalize_gait_cycle=True,
        save_path=output_dir / "comparison_ground_forces.png"
    )
    
    # Export results to CSV files
    print("Exporting results...")
    simulator.export_results(output_dir / "cot_simulation")
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main() 