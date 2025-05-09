"""
Simulation utilities for running batch experiments.

This module provides utilities for running batch simulations with different
cost functions and assistive device configurations.
"""

import json
import os
from typing import Dict, List, Optional
import numpy as np
from opensim import (
    Model,
    MocoTrack,
    MocoCasADiSolver,
    MocoStudy,
    MocoControlGoal,
    MocoStateTrackingGoal
)

def load_model(model_path: str) -> Model:
    """Load an OpenSim model from file.
    
    Args:
        model_path: Path to the .osim model file
        
    Returns:
        Loaded OpenSim model
    """
    model = Model(model_path)
    model.initSystem()
    return model

def create_tracking_study(model: Model, 
                         cost_function: str,
                         tracking_data=None,
                         time_interval: tuple = (0.0, 1.0)) -> MocoStudy:
    """Create a MocoTrack study with the specified cost function.
    
    Args:
        model: OpenSim model
        cost_function: Name of cost function to use
        tracking_data: Reference data to track (optional)
        time_interval: Tuple of (start_time, end_time)
        
    Returns:
        Configured MocoTrack study
    """
    # Create and name the tracking study
    study = MocoTrack()
    study.setName(f"tracking_{cost_function}")
    
    # Set the model
    study.setModel(model)
    
    # Set tracking reference if provided
    if tracking_data is not None:
        study.setStatesReference(tracking_data)
    
    # Add goals based on the cost function
    problem = study.updProblem()
    
    # Add control effort minimization
    effort = problem.addGoal(MocoControlGoal("effort"))
    effort.setWeight(0.1)
    
    # Add state tracking
    tracking = problem.addGoal(MocoStateTrackingGoal("tracking"))
    tracking.setWeight(1.0)
    
    # Configure the solver
    solver = study.initCasADiSolver()
    solver.set_num_mesh_intervals(50)  # Reduced for initial testing
    solver.set_optim_convergence_tolerance(1e-3)
    solver.set_optim_constraint_tolerance(1e-3)
    
    return study

def add_device(model: Model, location: str) -> None:
    """Add an assistive device to the model.
    
    Args:
        model: OpenSim model to modify
        location: Device location ("hip", "knee", or "ankle")
    """
    if location == "hip":
        joint_name = "hip_flexion_r"
    elif location == "knee":
        joint_name = "knee_angle_r"
    elif location == "ankle":
        joint_name = "ankle_angle_r"
    else:
        raise ValueError(f"Unknown device location: {location}")
    
    # Get the coordinate to assist
    coord = model.getCoordinateSet().get(joint_name)
    if not coord:
        raise ValueError(f"Could not find coordinate {joint_name}")
    
    # Create ideal torque actuator
    actuator = IdealTorqueActuator()
    actuator.setName(f"{location}_assist")
    actuator.setCoordinate(coord)
    actuator.setOptimalForce(100.0)  # N-m
    
    # Add to model
    model.addForce(actuator)
    model.initSystem()

def load_results(results_dir: str) -> Dict:
    """Load simulation results from directory.
    
    Args:
        results_dir: Directory containing .sto result files
        
    Returns:
        Dictionary mapping (cost_function, device_location) to solution
    """
    results = {}
    
    for filename in os.listdir(results_dir):
        if filename.endswith(".sto"):
            # Parse filename to get cost function and device location
            cost_func, device_loc = filename[:-4].split("_")
            
            # Load solution
            solution_path = os.path.join(results_dir, filename)
            try:
                solution = MocoTrajectory(solution_path)
                results[(cost_func, device_loc)] = solution
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    return results 