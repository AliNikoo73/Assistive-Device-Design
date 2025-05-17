"""
Tracking simulation module for GaitSim Assist.

This module provides functionality for running tracking simulations that follow
reference data, such as experimental gait data.
"""

import numpy as np
import opensim as osim
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd

from .gait_simulator import SimulationResults


class TrackingSimulation:
    """Class for running tracking simulations."""
    
    def __init__(self, model: osim.Model):
        """Initialize the tracking simulation.
        
        Args:
            model: OpenSim model to use for the simulation
        """
        self.model = model
        self.reference_data = None
        self.time_range = (0.0, 1.0)
        self.time_step = 0.01
        self.tracking_weight = 1.0
        self.effort_weight = 0.1
    
    def set_reference_data(self, reference_data: Union[str, Path, pd.DataFrame]) -> None:
        """Set the reference data for tracking.
        
        Args:
            reference_data: Path to reference data file or DataFrame with reference data
        """
        if isinstance(reference_data, (str, Path)):
            self.reference_data = pd.read_csv(reference_data)
        else:
            self.reference_data = reference_data
    
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
    
    def set_weights(self, tracking_weight: float = 1.0, effort_weight: float = 0.1) -> None:
        """Set the weights for tracking and effort terms in the cost function.
        
        Args:
            tracking_weight: Weight for tracking term
            effort_weight: Weight for effort term
        """
        self.tracking_weight = tracking_weight
        self.effort_weight = effort_weight
    
    def solve(self) -> Dict:
        """Solve the tracking problem.
        
        Returns:
            Solution dictionary
        """
        # Check if reference data is set
        if self.reference_data is None:
            raise ValueError("Reference data must be set before solving")
        
        # Create a MocoTrack study
        track = osim.MocoTrack()
        track.setName("tracking_problem")
        track.setModel(osim.ModelProcessor(self.model))
        
        # Set the reference data
        table = self._create_reference_table()
        track.setStatesReference(osim.TableProcessor(table))
        track.set_states_global_tracking_weight(self.tracking_weight)
        
        # Set the time range
        track.set_initial_time(self.time_range[0])
        track.set_final_time(self.time_range[1])
        
        # Set the solver
        solver = track.initCasADiSolver()
        solver.set_num_mesh_intervals(int((self.time_range[1] - self.time_range[0]) / self.time_step))
        solver.set_optim_convergence_tolerance(1e-4)
        solver.set_optim_constraint_tolerance(1e-4)
        
        # Set the effort term
        effort = osim.MocoControlGoal("effort")
        effort.setWeight(self.effort_weight)
        track.addGoal(effort)
        
        # Solve the problem
        solution = track.solve()
        
        # Return the solution as a dictionary
        return {
            'success': solution.success(),
            'solver_duration': solution.getSolverDuration(),
            'objective': solution.getObjective(),
            'solution': solution
        }
    
    def _create_reference_table(self) -> osim.TimeSeriesTable:
        """Create a reference table from the reference data.
        
        Returns:
            OpenSim TimeSeriesTable
        """
        # Create a TimeSeriesTable
        table = osim.TimeSeriesTable()
        
        # Add time column
        time_col = self.reference_data['time'].values
        
        # Add joint angle columns
        labels = []
        for joint in ['hip', 'knee', 'ankle']:
            if f'{joint}_angle' in self.reference_data.columns:
                labels.append(f'{joint}_angle')
        
        # Create the table
        for i, t in enumerate(time_col):
            row = osim.RowVector(len(labels))
            for j, label in enumerate(labels):
                row[j] = self.reference_data[label].values[i]
            table.appendRow(t, row)
        
        # Set the column labels
        table.setColumnLabels(labels)
        
        return table
    
    def analyze_results(self, solution_dict: Dict) -> SimulationResults:
        """Analyze the results of the tracking simulation.
        
        Args:
            solution_dict: Solution dictionary from solve()
            
        Returns:
            SimulationResults object
        """
        if not solution_dict['success']:
            raise RuntimeError("Tracking simulation failed")
        
        solution = solution_dict['solution']
        
        # Get the states and controls
        states_table = solution.exportToStatesTable()
        controls_table = solution.exportToControlsTable()
        
        # Extract time
        time = np.array([states_table.getIndependentColumn()[i] 
                       for i in range(states_table.getNumRows())])
        
        # Extract states
        states = {}
        for col in range(states_table.getNumColumns()):
            name = states_table.getColumnLabel(col)
            states[name] = np.array([states_table.getMatrix().getElt(row, col) 
                                   for row in range(states_table.getNumRows())])
        
        # Extract controls
        controls = {}
        for col in range(controls_table.getNumColumns()):
            name = controls_table.getColumnLabel(col)
            controls[name] = np.array([controls_table.getMatrix().getElt(row, col) 
                                     for row in range(controls_table.getNumRows())])
        
        # Extract joint angles
        joint_angles = {}
        for joint in ['hip', 'knee', 'ankle']:
            state_name = f'{joint}/flexion/value'
            if state_name in states:
                joint_angles[joint] = states[state_name]
        
        # Extract ground forces (simplified)
        ground_forces = {
            'vertical': np.zeros_like(time),
            'horizontal': np.zeros_like(time)
        }
        
        # Extract muscle activations
        muscle_activations = {}
        for name in states:
            if name.endswith('/activation'):
                muscle_name = name.split('/')[0]
                muscle_activations[muscle_name] = states[name]
        
        # Calculate metabolic cost (simplified)
        metabolic_cost = 100.0  # J/kg, placeholder
        
        # Create metrics dictionary
        metrics = {
            'tracking_error': solution_dict['objective'],
            'solver_duration': solution_dict['solver_duration']
        }
        
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
        """Visualize the tracking simulation.
        
        Args:
            solution_dict: Solution dictionary from solve()
        """
        if not solution_dict['success']:
            raise RuntimeError("Tracking simulation failed")
        
        solution = solution_dict['solution']
        
        # Create a visualizer
        visualizer = osim.MocoTrajectoryViz(solution.getMocoTrajectory())
        visualizer.setModel(self.model)
        visualizer.setTimeLimits(self.time_range[0], self.time_range[1])
        
        # Show the visualizer
        visualizer.showViz() 