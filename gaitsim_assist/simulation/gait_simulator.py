"""
GaitSimulator: Core simulation class for GaitSim Assist.

This class provides a unified API for running various types of gait simulations,
including tracking simulations and predictive simulations with different cost functions.
"""

import os
import numpy as np
import opensim as osim
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
from dataclasses import dataclass

from ..cost_functions import CostFunction


@dataclass
class SimulationResults:
    """Container for simulation results."""
    time: np.ndarray
    states: Dict[str, np.ndarray]
    controls: Dict[str, np.ndarray]
    joint_angles: Dict[str, np.ndarray]
    ground_forces: Dict[str, np.ndarray]
    muscle_activations: Dict[str, np.ndarray]
    metabolic_cost: float
    metrics: Dict[str, Any]


class GaitSimulator:
    """Main class for running gait simulations."""
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """Initialize the gait simulator.
        
        Args:
            model_path: Path to an OpenSim model file (.osim). If None, a default
                2D walking model will be created.
        """
        self.model = None
        self.model_path = model_path
        self.study = None
        self.solution = None
        self.results = None
        
        # Load or create the model
        if model_path is not None:
            self.load_model(model_path)
        else:
            self.create_default_model()
    
    def load_model(self, model_path: Union[str, Path]) -> osim.Model:
        """Load an OpenSim model from a file.
        
        Args:
            model_path: Path to the OpenSim model file (.osim)
            
        Returns:
            The loaded OpenSim model
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = osim.Model(str(model_path))
        self.model_path = model_path
        return self.model
    
    def create_default_model(self) -> osim.Model:
        """Create a default 2D walking model.
        
        Returns:
            The created OpenSim model
        """
        # Create a new OpenSim model
        model = osim.Model()
        model.setName('walk2d')
        
        # Set gravity
        model.setGravity(osim.Vec3(0, -9.81, 0))
        
        # Create ground reference frame
        ground = model.getGround()
        
        # Add a pelvis body
        pelvis = osim.Body('pelvis', 11.777, osim.Vec3(0), 
                          osim.Inertia(0.0765, 0.0765, 0.0765))
        model.addBody(pelvis)
        
        # Create a free joint to connect the pelvis to ground
        pelvis_tx = osim.SliderJoint('pelvis_tx', ground, osim.Vec3(0), osim.Vec3(0),
                                   pelvis, osim.Vec3(0), osim.Vec3(0))
        pelvis_tx.updCoordinate().setName('pelvis_tx')
        model.addJoint(pelvis_tx)
        
        pelvis_ty = osim.SliderJoint('pelvis_ty', pelvis, osim.Vec3(0), osim.Vec3(0),
                                   ground, osim.Vec3(0), osim.Vec3(0, 1, 0))
        pelvis_ty.updCoordinate().setName('pelvis_ty')
        model.addJoint(pelvis_ty)
        
        pelvis_tz = osim.SliderJoint('pelvis_tz', pelvis, osim.Vec3(0), osim.Vec3(0),
                                   ground, osim.Vec3(0), osim.Vec3(0, 0, 1))
        pelvis_tz.updCoordinate().setName('pelvis_tz')
        model.addJoint(pelvis_tz)
        
        pelvis_rot = osim.PinJoint('pelvis_rot', pelvis, osim.Vec3(0), osim.Vec3(0),
                                 ground, osim.Vec3(0), osim.Vec3(1, 0, 0))
        pelvis_rot.updCoordinate().setName('pelvis_rot')
        model.addJoint(pelvis_rot)
        
        # Add thigh, shank, and foot segments
        thigh = osim.Body('thigh', 8.806, osim.Vec3(0), 
                         osim.Inertia(0.1, 0.1, 0.1))
        model.addBody(thigh)
        
        shank = osim.Body('shank', 3.411, osim.Vec3(0),
                         osim.Inertia(0.05, 0.05, 0.05))
        model.addBody(shank)
        
        foot = osim.Body('foot', 1.20, osim.Vec3(0),
                        osim.Inertia(0.01, 0.01, 0.01))
        model.addBody(foot)
        
        # Add hip, knee, and ankle joints
        hip = osim.PinJoint('hip', pelvis, osim.Vec3(0), osim.Vec3(0),
                           thigh, osim.Vec3(0, 0.5, 0), osim.Vec3(0))
        hip.updCoordinate().setName('hip_flexion')
        model.addJoint(hip)
        
        knee = osim.PinJoint('knee', thigh, osim.Vec3(0, -0.5, 0), osim.Vec3(0),
                            shank, osim.Vec3(0, 0.5, 0), osim.Vec3(0))
        knee.updCoordinate().setName('knee_flexion')
        model.addJoint(knee)
        
        ankle = osim.PinJoint('ankle', shank, osim.Vec3(0, -0.5, 0), osim.Vec3(0),
                             foot, osim.Vec3(-0.05, 0.1, 0), osim.Vec3(0))
        ankle.updCoordinate().setName('ankle_flexion')
        model.addJoint(ankle)
        
        # Add muscles and ground contact
        self._add_muscles(model)
        self._add_ground_contact(model)
        
        # Finalize connections
        model.finalizeConnections()
        
        self.model = model
        return model
    
    def _add_muscles(self, model: osim.Model) -> None:
        """Add muscles to the model."""
        # Define muscle parameters
        max_isometric_force = 4000  # N
        optimal_fiber_length = 0.10  # m
        tendon_slack_length = 0.15   # m
        pennation_angle = 0.0
        
        # Create hip flexor
        hip_flexor = osim.Thelen2003Muscle('hip_flexor_r', max_isometric_force,
                                         optimal_fiber_length, tendon_slack_length,
                                         pennation_angle)
        hip_flexor.addNewPathPoint('origin', model.getBodySet().get('pelvis'),
                                 osim.Vec3(0.0, 0.1, 0))
        hip_flexor.addNewPathPoint('insertion', model.getBodySet().get('thigh'),
                                 osim.Vec3(0.0, 0.4, 0))
        model.addForce(hip_flexor)
        
        # Create hip extensor
        hip_extensor = osim.Thelen2003Muscle('hip_extensor_r', max_isometric_force,
                                           optimal_fiber_length, tendon_slack_length,
                                           pennation_angle)
        hip_extensor.addNewPathPoint('origin', model.getBodySet().get('pelvis'),
                                   osim.Vec3(0.0, -0.1, 0))
        hip_extensor.addNewPathPoint('insertion', model.getBodySet().get('thigh'),
                                   osim.Vec3(-0.05, -0.3, 0))
        model.addForce(hip_extensor)
        
        # Add more muscles (knee, ankle) as needed
        # ...
    
    def _add_ground_contact(self, model: osim.Model) -> None:
        """Add ground contact to the model."""
        ground = model.getGround()
        foot = model.getBodySet().get("foot")
        
        # Create ground force
        ground_force = osim.HuntCrossleyForce()
        ground_force.setName('ground_force')
        ground_force.setStiffness(1e6)  # N/m
        ground_force.setDissipation(1.0)  # s/m
        ground_force.setStaticFriction(0.8)
        ground_force.setDynamicFriction(0.4)
        
        # Create contact half-space for ground
        contact_ground = osim.ContactHalfSpace(osim.Vec3(0), osim.Vec3(0, 0, -np.pi/2), 
                                             ground, "ground_contact")
        ground_force.addGeometry("ground_contact")
        
        # Add contact points to the foot
        heel_point = osim.ContactSphere(0.02, osim.Vec3(-0.05, -0.02, 0), 
                                      foot, "heel")
        toe_point = osim.ContactSphere(0.02, osim.Vec3(0.05, -0.02, 0), 
                                     foot, "toe")
        ground_force.addGeometry("heel")
        ground_force.addGeometry("toe")
        
        model.addForce(ground_force)
    
    def run_tracking_simulation(self, 
                              reference_data: Union[str, Path, pd.DataFrame],
                              time_range: Tuple[float, float] = (0.0, 1.0),
                              visualize: bool = False) -> SimulationResults:
        """Run a tracking simulation to follow reference data.
        
        Args:
            reference_data: Path to reference data file or DataFrame with reference data
            time_range: Time range for the simulation (start, end)
            visualize: Whether to visualize the simulation
            
        Returns:
            Simulation results
        """
        from .tracking import TrackingSimulation
        
        tracking_sim = TrackingSimulation(self.model)
        tracking_sim.set_reference_data(reference_data)
        tracking_sim.set_time_range(time_range)
        
        self.solution = tracking_sim.solve()
        self.results = tracking_sim.analyze_results(self.solution)
        
        if visualize:
            tracking_sim.visualize(self.solution)
            
        return self.results
    
    def run_predictive_simulation(self, 
                               cost_function: Union[str, CostFunction],
                               time_range: Tuple[float, float] = (0.0, 1.0),
                               assistive_device: Optional[Any] = None,
                               visualize: bool = False) -> SimulationResults:
        """Run a predictive simulation with a specified cost function.
        
        Args:
            cost_function: Cost function to use (name or instance)
            time_range: Time range for the simulation (start, end)
            assistive_device: Optional assistive device to include
            visualize: Whether to visualize the simulation
            
        Returns:
            Simulation results
        """
        from .predictive import PredictiveSimulation
        from ..cost_functions import get_cost_function
        
        # Get the cost function
        if isinstance(cost_function, str):
            cost_function = get_cost_function(cost_function, self.model)
        
        # Create and run the predictive simulation
        predictive_sim = PredictiveSimulation(self.model)
        predictive_sim.set_cost_function(cost_function)
        predictive_sim.set_time_range(time_range)
        
        if assistive_device is not None:
            predictive_sim.add_assistive_device(assistive_device)
        
        self.solution = predictive_sim.solve()
        self.results = predictive_sim.analyze_results(self.solution)
        
        if visualize:
            predictive_sim.visualize(self.solution)
            
        return self.results
    
    def export_results(self, output_dir: Union[str, Path]) -> Path:
        """Export simulation results to files.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Path to the output directory
        """
        if self.results is None:
            raise ValueError("No simulation results to export. Run a simulation first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Export state trajectories
        states_df = pd.DataFrame({'time': self.results.time})
        for name, values in self.results.states.items():
            states_df[name] = values
        states_df.to_csv(output_dir / 'states.csv', index=False)
        
        # Export joint angles
        angles_df = pd.DataFrame({'time': self.results.time})
        for name, values in self.results.joint_angles.items():
            angles_df[name] = values
        angles_df.to_csv(output_dir / 'joint_angles.csv', index=False)
        
        # Export ground forces
        forces_df = pd.DataFrame({'time': self.results.time})
        for name, values in self.results.ground_forces.items():
            forces_df[name] = values
        forces_df.to_csv(output_dir / 'ground_forces.csv', index=False)
        
        # Export muscle activations
        activations_df = pd.DataFrame({'time': self.results.time})
        for name, values in self.results.muscle_activations.items():
            activations_df[name] = values
        activations_df.to_csv(output_dir / 'muscle_activations.csv', index=False)
        
        # Export metrics summary
        metrics_df = pd.DataFrame(self.results.metrics, index=[0])
        metrics_df.to_csv(output_dir / 'metrics.csv', index=False)
        
        return output_dir 