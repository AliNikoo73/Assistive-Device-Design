"""
Muscle effort cost function for GaitSim Assist.

This module provides a cost function that minimizes muscle effort,
defined as the sum of squared muscle activations.
"""

import numpy as np
import opensim as osim
import opensim.moco as moco
from typing import Dict, List, Optional, Union, Any

from .base import CostFunction


class MuscleEffort(CostFunction):
    """Cost function that minimizes muscle effort."""
    
    def __init__(self, model: osim.Model, weight: float = 1.0):
        """Initialize the muscle effort cost function.
        
        Args:
            model: OpenSim model
            weight: Weight for this cost function
        """
        super().__init__(model, weight)
        self.name = "muscle_effort"
        
        # Get list of muscles in the model
        self.muscles = []
        for i in range(model.getMuscles().getSize()):
            self.muscles.append(model.getMuscles().get(i).getName())
    
    def implement(self) -> moco.MocoGoal:
        """Implement the cost function as a Moco goal.
        
        Returns:
            Moco goal for muscle effort
        """
        # Create a sum of squared controls cost
        effort = moco.MocoControlGoal("muscle_effort")
        
        # Add all muscle controls
        for muscle_name in self.muscles:
            effort.addControlPath(f"/{muscle_name}/activation")
        
        # Set the weight
        effort.setWeight(self.weight)
        
        return effort
    
    def compute(self, states: Dict[str, np.ndarray], 
               controls: Dict[str, np.ndarray], 
               time: np.ndarray) -> float:
        """Compute the muscle effort cost for given states and controls.
        
        Args:
            states: Dictionary of state trajectories
            controls: Dictionary of control trajectories
            time: Time points
            
        Returns:
            Total muscle effort cost
        """
        total_effort = 0.0
        
        # Sum squared muscle activations
        for muscle_name in self.muscles:
            control_name = f"{muscle_name}/activation"
            if control_name in controls:
                total_effort += np.sum(np.square(controls[control_name]))
        
        # Apply weight
        total_effort *= self.weight
        
        return total_effort 