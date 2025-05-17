"""
Predictive simulation module for GaitSim Assist.

This module provides functionality for running predictive simulations that
optimize a cost function to predict gait patterns.
"""

import numpy as np
import opensim as osim
import opensim.moco as moco
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd

from .gait_simulator import SimulationResults
from ..cost_functions import CostFunction


class PredictiveSimulation:
    """Class for running predictive simulations."""
    
    def __init__(self, model: osim.Model):
        """Initialize the predictive simulation.
        
        Args:
            model: OpenSim model to use for the simulation
        """
        self.model = model
        self.cost_function = None
        self.time_range = (0.0, 1.0)
        self.time_step = 0.01
        self.assistive_device = None
    
    def set_cost_function(self, cost_function: CostFunction) -> None:
        """Set the cost function for optimization.
        
        Args:
            cost_function: Cost function to optimize
        """
        self.cost_function = cost_function
    
    def set_time_range(self, time_range: Tuple[float, float]) -> None:
        """Set the time range for the simulation.
        
        Args:
            time_range: Time range (start, end) in seconds
        """
        self.time_range = time_range
    
    def set_time_step(self, time_step: float) -> None:
        """Set the time step for the simulation.
        
        Args:
            time_step: Time step in seconds
        """
        self.time_step = time_step
    
    def add_assistive_device(self, device) -> None:
        """Add an assistive device to the simulation.
        
        Args:
            device: Assistive device to add
        """
        self.assistive_device = device
        
        # Apply the device to the model
        device.apply_to_model()
    
    def solve(self) -> Dict:
        """Solve the predictive problem using OpenSim Moco.
        
        Returns:
            Solution dictionary
        """
        # Check if cost function is set
        if self.cost_function is None:
            raise ValueError("Cost function must be set before solving")
        
        # Create a Moco study
        study = moco.MocoStudy()
        study.setName("predictive_study")
        
        # Create a MocoProblem
        problem = study.updProblem()
        
        # Set the model
        model_processor = osim.ModelProcessor(self.model)
        problem.setModelProcessor(model_processor)
        
        # Add the cost function goal
        cost_goal = moco.MocoGoal.safeDownCast(problem.addGoal(moco.MocoCustomGoal("cost_function", self.cost_function)))
        cost_goal.setRequirements(0, 0, moco.Stage.Dynamics)
        
        # Add a periodic constraint for cyclic gait
        periodic = moco.MocoPeriodicConstraint()
        periodic.setName("periodic_constraint")
        
        # Add states to make periodic (position and velocity for each joint)
        for joint in ['hip', 'knee', 'ankle']:
            periodic.addStatePair(f"{joint}/flexion/value")
            periodic.addStatePair(f"{joint}/flexion/speed")
        
        problem.addPathConstraint(periodic)
        
        # Configure the solver
        solver = study.initSolver(moco.MocoCasADiSolver())
        solver.set_num_mesh_intervals(int((self.time_range[1] - self.time_range[0]) / self.time_step))
        solver.set_optim_convergence_tolerance(1e-4)
        solver.set_optim_constraint_tolerance(1e-4)
        solver.set_multibody_dynamics_mode("implicit")
        solver.set_minimize_implicit_auxiliary_derivatives(True)
        
        # Set time range
        solver.set_initial_time(self.time_range[0])
        solver.set_final_time(self.time_range[1])
        
        # Set bounds on states and controls
        # (simplified for brevity)
        
        # Solve the problem
        solution = study.solve()
        
        # Return the solution as a dictionary
        return {
            'success': solution.success(),
            'solver_duration': solution.getSolverDuration(),
            'objective': solution.getObjective(),
            'solution': solution
        }
    
    def analyze_results(self, solution_dict: Dict) -> SimulationResults:
        """Analyze the results of the predictive simulation.
        
        Args:
            solution_dict: Solution dictionary from solve()
            
        Returns:
            SimulationResults object
        """
        if not solution_dict['success']:
            raise RuntimeError("Predictive simulation failed")
        
        solution = solution_dict['solution']
        
        # Get the states and controls
        states_table = solution.exportToStatesTable()
        controls_table = solution.exportToControlsTable()
        
        # Extract time
        time = np.array(states_table.getIndependentColumn())
        
        # Extract states
        states = {}
        for col in range(states_table.getNumColumns()):
            name = states_table.getColumnLabel(col)
            states[name] = np.array(states_table.getDependentColumnAtIndex(col))
        
        # Extract controls
        controls = {}
        for col in range(controls_table.getNumColumns()):
            name = controls_table.getColumnLabel(col)
            controls[name] = np.array(controls_table.getDependentColumnAtIndex(col))
        
        # Extract joint angles
        joint_angles = {}
        for joint in ['hip', 'knee', 'ankle']:
            state_name = f'{joint}/flexion/value'
            if state_name in states:
                joint_angles[joint] = states[state_name]
        
        # Extract ground forces
        ground_forces = {
            'vertical': np.zeros_like(time),
            'horizontal': np.zeros_like(time)
        }
        
        # Try to get ground reaction forces from contact forces
        try:
            contact_forces = solution.exportToContactForceTable()
            if contact_forces.getNumColumns() > 0:
                for col in range(contact_forces.getNumColumns()):
                    name = contact_forces.getColumnLabel(col)
                    if 'vertical' in name.lower():
                        ground_forces['vertical'] = np.array(contact_forces.getDependentColumnAtIndex(col))
                    elif 'horizontal' in name.lower() or 'anterior' in name.lower():
                        ground_forces['horizontal'] = np.array(contact_forces.getDependentColumnAtIndex(col))
        except Exception as e:
            print(f"Could not extract contact forces: {e}")
        
        # Extract muscle activations
        muscle_activations = {}
        for name in states:
            if name.endswith('/activation'):
                muscle_name = name.split('/')[0]
                muscle_activations[muscle_name] = states[name]
        
        # Calculate metabolic cost using the cost function
        if hasattr(self.cost_function, 'compute'):
            metabolic_cost = self.cost_function.compute(states, controls, time)
        else:
            metabolic_cost = solution_dict['objective']  # Use objective as a proxy
        
        # Create metrics dictionary
        metrics = {
            'objective': solution_dict['objective'],
            'solver_duration': solution_dict['solver_duration']
        }
        
        # Add assistive device metrics if present
        if self.assistive_device is not None:
            metrics['device_mass'] = self.assistive_device.get_mass()
            metrics['device_parameters'] = self.assistive_device.get_parameters()
        
        # Create and return SimulationResults
        return SimulationResults(
            time=time,
            states=states,
            controls=controls,
            joint_angles=joint_angles,
            ground_forces=ground_forces,
            muscle_activations=muscle_activations,
            metabolic_cost=metabolic_cost,
            metrics=metrics
        )
    
    def visualize(self, solution_dict: Dict) -> None:
        """Visualize the predictive simulation.
        
        Args:
            solution_dict: Solution dictionary from solve()
        """
        if not solution_dict['success']:
            raise RuntimeError("Predictive simulation failed")
        
        solution = solution_dict['solution']
        
        # Create a visualizer
        visualizer = moco.MocoTrajectoryViz(solution.unsealedMocoTrajectory())
        visualizer.setModel(self.model)
        visualizer.setTimeLimits(self.time_range[0], self.time_range[1])
        
        # Show the visualizer
        visualizer.showViz() 