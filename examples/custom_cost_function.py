#!/usr/bin/env python3
"""
Example of creating and using a custom cost function with GaitSim Assist.

This example demonstrates how to:
1. Create a custom cost function by inheriting from the CostFunction base class
2. Use the custom cost function in a predictive simulation
3. Visualize the results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import gaitsim_assist as gsa
from gaitsim_assist.cost_functions import CostFunction
from gaitsim_assist.visualization import GaitPlotter


class CustomCostFunction(CostFunction):
    """Custom cost function that minimizes joint jerk.
    
    This cost function minimizes the third derivative of joint angles (jerk),
    which promotes smooth motion.
    """
    
    def __init__(self, model, weight=1.0, name=None):
        """Initialize the cost function.
        
        Args:
            model: OpenSim model
            weight: Weight of this cost function
            name: Name of this cost function
        """
        super().__init__(model, weight, name or "JointJerk")
    
    def compute(self, states, controls, time):
        """Compute the cost function value.
        
        Args:
            states: State variables
            controls: Control variables
            time: Time points
            
        Returns:
            Cost function value
        """
        # Extract joint angles
        joint_angles = {}
        for joint in ['hip', 'knee', 'ankle']:
            state_name = f'{joint}/flexion/value'
            if state_name in states:
                joint_angles[joint] = states[state_name]
        
        # Calculate jerk for each joint
        dt = time[1] - time[0] if len(time) > 1 else 1.0
        total_jerk = 0.0
        
        for joint, angles in joint_angles.items():
            # Calculate first derivative (velocity)
            velocity = np.gradient(angles, dt)
            
            # Calculate second derivative (acceleration)
            acceleration = np.gradient(velocity, dt)
            
            # Calculate third derivative (jerk)
            jerk = np.gradient(acceleration, dt)
            
            # Sum of squared jerk
            total_jerk += np.sum(jerk**2)
        
        return total_jerk


def main():
    # Create output directory
    output_dir = Path("simulation_results")
    output_dir.mkdir(exist_ok=True)
    
    print("Creating gait simulator...")
    # Create a gait simulator with default 2D walking model
    simulator = gsa.GaitSimulator()
    
    # Create a custom cost function
    print("Creating custom cost function...")
    custom_cost = CustomCostFunction(simulator.model)
    
    # Run a predictive simulation with the custom cost function
    print("Running simulation with custom cost function...")
    custom_results = simulator.run_predictive_simulation(
        cost_function=custom_cost,
        time_range=(0.0, 1.0)
    )
    
    # Run a predictive simulation with the default cost of transport cost function
    print("Running simulation with cost of transport...")
    cot_results = simulator.run_predictive_simulation(
        cost_function='cot',
        time_range=(0.0, 1.0)
    )
    
    # Visualize the results
    print("Visualizing results...")
    plotter = GaitPlotter()
    
    # Compare joint angles between different simulations
    plotter.compare_simulations(
        results_list=[custom_results, cot_results],
        labels=["Joint Jerk", "Cost of Transport"],
        plot_type="joint_angles",
        normalize_gait_cycle=True,
        save_path=output_dir / "comparison_joint_angles.png"
    )
    
    # Compare ground reaction forces between different simulations
    plotter.compare_simulations(
        results_list=[custom_results, cot_results],
        labels=["Joint Jerk", "Cost of Transport"],
        plot_type="ground_forces",
        normalize_gait_cycle=True,
        save_path=output_dir / "comparison_ground_forces.png"
    )
    
    # Export results to CSV files
    print("Exporting results...")
    simulator.export_results(output_dir / "custom_cost_simulation")
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main() 