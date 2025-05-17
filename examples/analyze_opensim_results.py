#!/usr/bin/env python3
"""
Example script to analyze existing OpenSim simulation results.

This example demonstrates how to:
1. Load OpenSim simulation results
2. Convert them to GaitSim Assist format
3. Analyze and visualize the results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import opensim as osim

import gaitsim_assist as gsa
from gaitsim_assist.visualization import GaitPlotter
from gaitsim_assist.analysis import calculate_gait_metrics

# Create output directory
output_dir = Path("opensim_analysis_results")
output_dir.mkdir(exist_ok=True)


def load_opensim_states(file_path):
    """Load OpenSim states from a storage file."""
    table = osim.TimeSeriesTable(str(file_path))
    time = np.array(table.getIndependentColumn())
    
    # Extract states
    states = {}
    for col in range(table.getNumColumns()):
        name = table.getColumnLabel(col)
        states[name] = np.array(table.getDependentColumnAtIndex(col))
    
    return time, states


def load_opensim_forces(file_path):
    """Load OpenSim forces from a storage file."""
    table = osim.TimeSeriesTable(str(file_path))
    time = np.array(table.getIndependentColumn())
    
    # Extract forces
    forces = {}
    for col in range(table.getNumColumns()):
        name = table.getColumnLabel(col)
        forces[name] = np.array(table.getDependentColumnAtIndex(col))
    
    return time, forces


def convert_to_gaitsim_results(states_file, forces_file=None, controls_file=None):
    """Convert OpenSim results to GaitSim Assist format."""
    # Load states
    time, states = load_opensim_states(states_file)
    
    # Extract joint angles
    joint_angles = {}
    for name in states:
        if '/value' in name:
            joint_name = name.split('/')[0]
            if any(j in joint_name for j in ['hip', 'knee', 'ankle']):
                joint_angles[joint_name] = states[name]
    
    # Load forces if available
    ground_forces = {
        'vertical': np.zeros_like(time),
        'horizontal': np.zeros_like(time)
    }
    
    if forces_file is not None:
        _, forces = load_opensim_forces(forces_file)
        
        # Extract ground reaction forces
        for name in forces:
            if 'ground_force_vy' in name:
                ground_forces['vertical'] = forces[name]
            elif 'ground_force_vx' in name:
                ground_forces['horizontal'] = forces[name]
    
    # Load controls if available
    controls = {}
    muscle_activations = {}
    
    if controls_file is not None:
        _, control_data = load_opensim_states(controls_file)
        controls = control_data
        
        # Extract muscle activations
        for name in control_data:
            if 'activation' in name:
                muscle_name = name.split('/')[0]
                muscle_activations[muscle_name] = control_data[name]
    
    # Calculate basic metrics
    metrics = {
        'stride_length': 0.0,
        'stride_time': time[-1] - time[0],
        'cadence': 0.0
    }
    
    # If pelvis translation is available, calculate stride length
    if 'pelvis_tx/value' in states:
        metrics['stride_length'] = states['pelvis_tx/value'][-1] - states['pelvis_tx/value'][0]
        metrics['cadence'] = 60.0 / metrics['stride_time']  # steps per minute
    
    # Create simulation results
    results = gsa.simulation.SimulationResults(
        time=time,
        states=states,
        controls=controls,
        joint_angles=joint_angles,
        ground_forces=ground_forces,
        muscle_activations=muscle_activations,
        metabolic_cost=0.0,  # Placeholder
        metrics=metrics
    )
    
    return results


def analyze_results(results):
    """Analyze gait simulation results."""
    # Calculate comprehensive gait metrics
    metrics = calculate_gait_metrics(results)
    
    # Print key metrics
    print("\nGait Analysis Results:")
    print("-" * 40)
    print(f"Stride Length: {metrics.get('stride_length', 0):.3f} m")
    print(f"Stride Time: {metrics.get('stride_time', 0):.3f} s")
    print(f"Cadence: {metrics.get('cadence', 0):.1f} steps/min")
    print(f"Walking Speed: {metrics.get('walking_speed', 0):.2f} m/s")
    
    if 'knee_rom' in metrics:
        print(f"Knee Range of Motion: {metrics.get('knee_rom', 0):.1f} degrees")
    
    if 'peak_grf' in metrics:
        print(f"Peak Ground Reaction Force: {metrics.get('peak_grf', 0):.1f} N")
    
    return metrics


def visualize_results(results, output_dir):
    """Visualize gait simulation results."""
    # Create a plotter
    plotter = GaitPlotter()
    
    # Plot joint angles
    joint_angles_fig = plotter.plot_joint_angles(
        results,
        normalize_gait_cycle=True,
        save_path=output_dir / "joint_angles.png"
    )
    
    # Plot ground reaction forces
    grf_fig = plotter.plot_ground_forces(
        results,
        normalize_gait_cycle=True,
        save_path=output_dir / "ground_forces.png"
    )
    
    # Plot muscle activations if available
    if results.muscle_activations:
        activations_fig = plotter.plot_muscle_activations(
            results,
            normalize_gait_cycle=True,
            save_path=output_dir / "muscle_activations.png"
        )


def main():
    # Look for OpenSim result files
    opensim_dir = Path("opensim_results")
    
    if not opensim_dir.exists():
        print(f"Directory {opensim_dir} not found. Creating example directory structure.")
        opensim_dir.mkdir(exist_ok=True)
        print(f"Please place OpenSim result files in {opensim_dir} directory.")
        print("Expected files: states.sto, forces.sto, controls.sto")
        return
    
    # Find result files
    states_file = opensim_dir / "states.sto"
    forces_file = opensim_dir / "forces.sto"
    controls_file = opensim_dir / "controls.sto"
    
    if not states_file.exists():
        print(f"States file {states_file} not found.")
        print("Please provide an OpenSim states file to analyze.")
        return
    
    # Convert OpenSim results to GaitSim format
    print(f"Loading OpenSim results from {opensim_dir}...")
    results = convert_to_gaitsim_results(
        states_file,
        forces_file if forces_file.exists() else None,
        controls_file if controls_file.exists() else None
    )
    
    # Analyze the results
    print("Analyzing results...")
    metrics = analyze_results(results)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "gait_metrics.csv", index=False)
    
    # Visualize the results
    print("Visualizing results...")
    visualize_results(results, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main() 