#!/usr/bin/env python3
"""
Example of batch processing multiple simulations with GaitSim Assist.

This example demonstrates how to:
1. Run multiple simulations with different cost functions
2. Process the results in batch
3. Generate comparative visualizations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any

import gaitsim_assist as gsa
from gaitsim_assist.visualization import GaitPlotter
from gaitsim_assist.analysis import calculate_gait_metrics


def run_simulation(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single simulation with the given configuration.
    
    Args:
        config: Simulation configuration
        
    Returns:
        Dictionary with simulation results
    """
    # Create a simulator
    simulator = gsa.GaitSimulator()
    
    # Extract configuration parameters
    cost_function_name = config.get('cost_function', 'cot')
    time_range = tuple(config.get('time_range', (0.0, 1.0)))
    
    # Check if we need to add an assistive device
    assistive_device = None
    if 'device' in config:
        device_config = config['device']
        device_type = device_config.get('type', 'exoskeleton')
        
        if device_type == 'exoskeleton':
            assistive_device = gsa.devices.Exoskeleton(
                name=device_config.get('name', 'exo'),
                model=simulator.model,
                joint_name=device_config.get('joint_name', 'ankle'),
                mass=device_config.get('mass', 1.0),
                max_torque=device_config.get('max_torque', 50.0)
            )
    
    # Run the simulation
    results = simulator.run_predictive_simulation(
        cost_function=cost_function_name,
        time_range=time_range,
        assistive_device=assistive_device
    )
    
    # Calculate metrics
    metrics = calculate_gait_metrics(results)
    
    # Return the results and metrics
    return {
        'config': config,
        'results': results,
        'metrics': metrics
    }


def main():
    # Create output directory
    output_dir = Path("simulation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Define simulation configurations
    configurations = [
        {
            'name': 'cot',
            'cost_function': 'cot',
            'time_range': [0.0, 1.0],
            'description': 'Cost of transport optimization'
        },
        {
            'name': 'muscle_effort',
            'cost_function': 'muscle_effort',
            'time_range': [0.0, 1.0],
            'description': 'Muscle effort optimization'
        },
        {
            'name': 'fatigue',
            'cost_function': 'fatigue',
            'time_range': [0.0, 1.0],
            'description': 'Fatigue optimization'
        },
        {
            'name': 'ankle_exo',
            'cost_function': 'cot',
            'time_range': [0.0, 1.0],
            'description': 'Cost of transport with ankle exoskeleton',
            'device': {
                'type': 'exoskeleton',
                'name': 'ankle_exo',
                'joint_name': 'ankle',
                'mass': 1.0,
                'max_torque': 50.0
            }
        },
        {
            'name': 'knee_exo',
            'cost_function': 'cot',
            'time_range': [0.0, 1.0],
            'description': 'Cost of transport with knee exoskeleton',
            'device': {
                'type': 'exoskeleton',
                'name': 'knee_exo',
                'joint_name': 'knee',
                'mass': 1.0,
                'max_torque': 50.0
            }
        }
    ]
    
    # Save configurations to a JSON file
    with open(output_dir / "configurations.json", 'w') as f:
        json.dump(configurations, f, indent=2)
    
    # Run simulations in parallel
    print(f"Running {len(configurations)} simulations in parallel...")
    results = []
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit all simulations
        future_to_config = {executor.submit(run_simulation, config): config 
                          for config in configurations}
        
        # Process results as they complete
        for i, future in enumerate(future_to_config):
            config = future_to_config[future]
            print(f"Processing simulation {i+1}/{len(configurations)}: {config['name']}...")
            
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Simulation {config['name']} failed: {e}")
    
    # Extract metrics from all simulations
    all_metrics = []
    for result in results:
        metrics = result['metrics']
        metrics['name'] = result['config']['name']
        metrics['description'] = result['config']['description']
        all_metrics.append(metrics)
    
    # Create a DataFrame with all metrics
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save metrics to a CSV file
    metrics_df.to_csv(output_dir / "all_metrics.csv", index=False)
    
    # Create visualization for comparison
    print("Creating visualizations...")
    plotter = GaitPlotter()
    
    # Extract simulation results
    sim_results = [r['results'] for r in results]
    sim_names = [r['config']['name'] for r in results]
    
    # Compare joint angles
    plotter.compare_simulations(
        results_list=sim_results,
        labels=sim_names,
        plot_type="joint_angles",
        normalize_gait_cycle=True,
        save_path=output_dir / "batch_comparison_joint_angles.png"
    )
    
    # Compare ground forces
    plotter.compare_simulations(
        results_list=sim_results,
        labels=sim_names,
        plot_type="ground_forces",
        normalize_gait_cycle=True,
        save_path=output_dir / "batch_comparison_ground_forces.png"
    )
    
    # Create a bar chart comparing metabolic cost
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_df['name'], metrics_df['metabolic_cost'])
    plt.ylabel('Metabolic Cost (J/kg/m)')
    plt.title('Metabolic Cost Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "metabolic_cost_comparison.png", dpi=300)
    
    # Create a radar chart comparing multiple metrics
    metrics_to_plot = ['metabolic_cost', 'step_length', 'cadence', 'walking_speed']
    
    # Normalize the metrics for the radar chart
    normalized_metrics = metrics_df[metrics_to_plot].copy()
    for col in normalized_metrics.columns:
        normalized_metrics[col] = (normalized_metrics[col] - normalized_metrics[col].min()) / \
                                (normalized_metrics[col].max() - normalized_metrics[col].min())
    
    # Create the radar chart
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Set the angles for each metric
    angles = np.linspace(0, 2*np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Plot each simulation
    for i, name in enumerate(metrics_df['name']):
        values = normalized_metrics.iloc[i].tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.1)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot)
    ax.set_title('Normalized Metrics Comparison')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_radar_chart.png", dpi=300)
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main() 