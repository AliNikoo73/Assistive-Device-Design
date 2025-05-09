"""
Tracking simulation script.

This script uses MocoTrack to track healthy gait data with different
cost functions.
"""

import os
import json
import numpy as np
from opensim import (
    Model, 
    MocoTrack, 
    MocoCasADiSolver,
    Storage,
    TimeSeriesTable,
    StatesTrajectory,
    Vector
)
from utilities import load_model, create_tracking_study

def create_synthetic_data(model: Model, duration: float = 1.0) -> TimeSeriesTable:
    """Create synthetic motion data for tracking.
    
    Args:
        model: OpenSim model
        duration: Duration of the motion in seconds
        
    Returns:
        TimeSeriesTable with synthetic motion data
    """
    print("Creating synthetic motion data...")
    
    # Get coordinates
    coords = model.getCoordinateSet()
    num_coords = coords.getSize()
    
    # Create time vector (100 points)
    num_points = 100
    time = np.linspace(0, duration, num_points)
    
    # Create synthetic motion (simple sinusoidal)
    column_labels = []
    data_matrix = np.zeros((num_points, num_coords))
    
    for i in range(num_coords):
        coord = coords.get(i)
        column_labels.append(coord.getName())
        
        # Generate simple sinusoidal motion
        amplitude = np.deg2rad(10)  # 10 degrees
        freq = 1.0  # 1 Hz
        phase = np.random.rand() * 2 * np.pi  # Random phase
        
        # Create trajectory
        data_matrix[:, i] = amplitude * np.sin(2 * np.pi * freq * time + phase)
    
    # Create table
    table = TimeSeriesTable()
    table.setColumnLabels(column_labels)
    
    # Add data rows
    for i in range(num_points):
        row_vector = Vector(data_matrix[i, :].tolist())
        table.appendRow(float(time[i]), row_vector)
    
    print("Synthetic data created successfully!")
    return table

def main():
    # Configuration
    config = {
        "model_path": "../Models/Gait10dof18musc/gait10dof18musc.osim",  # Fixed path
        "data_path": "../data/healthy_gait/gait08",  # Fixed path
        "cost_functions": [
            "cot",
            "muscle_effort",
            "joint_torque",
            "fatigue",
            "head_motion",
            "hybrid"
        ],
        "output_dir": "../results/tracking"  # Fixed path
    }
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    print(f"Loading model from {config['model_path']}...")
    
    # Load model
    try:
        model = Model(config["model_path"])
        model.initSystem()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create synthetic tracking data
    try:
        tracking_data = create_synthetic_data(model)
        print("Tracking data created successfully!")
    except Exception as e:
        print(f"Error creating synthetic data: {e}")
        return
    
    # Run tracking for each cost function
    for cost_func in config["cost_functions"]:
        print(f"\nRunning tracking with {cost_func} cost function...")
        
        try:
            # Create tracking study
            study = create_tracking_study(model, cost_func, tracking_data)
            
            # Solve
            solution = study.solve()
            
            # Save results
            output_path = os.path.join(
                config["output_dir"],
                f"{cost_func}_tracking.sto"
            )
            solution.write(output_path)
            
            print(f"Saved results to {output_path}")
            
        except Exception as e:
            print(f"Error running {cost_func} tracking: {e}")
            continue

if __name__ == "__main__":
    main() 