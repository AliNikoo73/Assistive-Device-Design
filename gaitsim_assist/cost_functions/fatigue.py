"""
Fatigue cost function for GaitSim Assist.

This module provides a cost function that minimizes muscle fatigue,
defined as the maximum muscle activation.
"""

import numpy as np
import opensim as osim
import opensim.moco as moco
from typing import Dict, List, Optional, Union, Any

from .base import CostFunction


class Fatigue(CostFunction):
    """Cost function that minimizes muscle fatigue."""
    
    def __init__(self, model: osim.Model, weight: float = 1.0, exponent: float = 3.0):
        """Initialize the fatigue cost function.
        
        Args:
            model: OpenSim model
            weight: Weight for this cost function
            exponent: Exponent for the activation (higher values penalize high activations more)
        """
        super().__init__(model, weight)
        self.name = "fatigue"
        self.exponent = exponent
        
        # Get list of muscles in the model
        self.muscles = []
        for i in range(model.getMuscles().getSize()):
            self.muscles.append(model.getMuscles().get(i).getName())
    
    def implement(self) -> moco.MocoGoal:
        """Implement the cost function as a Moco goal.
        
        Returns:
            Moco goal for fatigue
        """
        # Create a control goal with high exponent to minimize peak activations
        fatigue = moco.MocoControlGoal("fatigue")
        
        # Add all muscle controls with the specified exponent
        for muscle_name in self.muscles:
            fatigue.addControlPath(f"/{muscle_name}/activation")
        
        # Set the exponent (higher values penalize high activations more)
        fatigue.setExponent(self.exponent)
        
        # Set the weight
        fatigue.setWeight(self.weight)
        
        return fatigue
    
    def compute(self, states: Dict[str, np.ndarray], 
               controls: Dict[str, np.ndarray], 
               time: np.ndarray) -> float:
        """Compute the fatigue cost for given states and controls.
        
        Args:
            states: Dictionary of state trajectories
            controls: Dictionary of control trajectories
            time: Time points
            
        Returns:
            Total fatigue cost
        """
        total_fatigue = 0.0
        
        # Compute maximum activation for each muscle
        for muscle_name in self.muscles:
            control_name = f"{muscle_name}/activation"
            if control_name in controls:
                # Sum of activations raised to the exponent
                total_fatigue += np.sum(np.power(controls[control_name], self.exponent))
        
        # Apply weight
        total_fatigue *= self.weight
        
        return total_fatigue 