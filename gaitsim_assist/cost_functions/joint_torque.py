"""
Joint torque cost function for GaitSim Assist.

This module provides a cost function that minimizes joint torques,
defined as the sum of squared joint torques.
"""

import numpy as np
import opensim as osim
import opensim.moco as moco
from typing import Dict, List, Optional, Union, Any

from .base import CostFunction


class JointTorque(CostFunction):
    """Cost function that minimizes joint torques."""
    
    def __init__(self, model: osim.Model, weight: float = 1.0):
        """Initialize the joint torque cost function.
        
        Args:
            model: OpenSim model
            weight: Weight for this cost function
        """
        super().__init__(model, weight)
        self.name = "joint_torque"
        
        # Get list of joints in the model
        self.joints = []
        for i in range(model.getJointSet().getSize()):
            joint = model.getJointSet().get(i)
            for j in range(joint.numCoordinates()):
                coord = joint.get_coordinates(j)
                self.joints.append(coord.getName())
    
    def implement(self) -> moco.MocoGoal:
        """Implement the cost function as a Moco goal.
        
        Returns:
            Moco goal for joint torque
        """
        # Create a sum of squared actuator controls cost
        torque = moco.MocoControlGoal("joint_torque")
        
        # Add all joint actuator controls
        for joint_name in self.joints:
            torque.addControlPath(f"/{joint_name}_actuator")
        
        # Set the weight
        torque.setWeight(self.weight)
        
        return torque
    
    def compute(self, states: Dict[str, np.ndarray], 
               controls: Dict[str, np.ndarray], 
               time: np.ndarray) -> float:
        """Compute the joint torque cost for given states and controls.
        
        Args:
            states: Dictionary of state trajectories
            controls: Dictionary of control trajectories
            time: Time points
            
        Returns:
            Total joint torque cost
        """
        total_torque = 0.0
        
        # Sum squared joint torques
        for joint_name in self.joints:
            control_name = f"{joint_name}_actuator"
            if control_name in controls:
                total_torque += np.sum(np.square(controls[control_name]))
        
        # Apply weight
        total_torque *= self.weight
        
        return total_torque 