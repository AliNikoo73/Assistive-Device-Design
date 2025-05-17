"""
Base cost function class for GaitSim Assist.

This module defines the base CostFunction class that all cost functions must inherit from.
"""

import numpy as np
import opensim as osim
import opensim.moco as moco
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union


class CostFunction(ABC):
    """Base class for all cost functions."""
    
    def __init__(self, model, weight: float = 1.0, name: Optional[str] = None):
        """Initialize the cost function.
        
        Args:
            model: OpenSim model
            weight: Weight of this cost function when combined with others
            name: Optional name for this cost function instance
        """
        self.model = model
        self.weight = weight
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def compute(self, states: Dict[str, np.ndarray], controls: Dict[str, np.ndarray], 
                time: np.ndarray) -> float:
        """Compute the cost function value.
        
        Args:
            states: State variables (positions, velocities, activations, etc.)
            controls: Control variables (excitations, torques, etc.)
            time: Time points
            
        Returns:
            Cost function value
        """
        pass
    
    def implement(self) -> moco.MocoGoal:
        """Implement the cost function as a Moco goal.
        
        This method should be overridden by subclasses to provide a Moco goal
        implementation of the cost function.
        
        Returns:
            A Moco goal representing this cost function
        """
        raise NotImplementedError(
            f"Cost function {self.name} does not implement the 'implement' method "
            "required for use with OpenSim Moco."
        )
    
    def __call__(self, states: Dict[str, np.ndarray], controls: Dict[str, np.ndarray], 
                time: np.ndarray) -> float:
        """Compute the weighted cost function value.
        
        Args:
            states: State variables
            controls: Control variables
            time: Time points
            
        Returns:
            Weighted cost function value
        """
        return self.weight * self.compute(states, controls, time) 