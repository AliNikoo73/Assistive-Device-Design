"""
Hybrid cost function for GaitSim Assist.

This module provides a cost function that combines multiple cost functions
with different weights.
"""

import numpy as np
import opensim as osim
import opensim.moco as moco
from typing import Dict, List, Optional, Union, Any

from .base import CostFunction


class Hybrid(CostFunction):
    """Cost function that combines multiple cost functions with weights."""
    
    def __init__(self, model: osim.Model, cost_functions: Dict[str, float], weight: float = 1.0):
        """Initialize the hybrid cost function.
        
        Args:
            model: OpenSim model
            cost_functions: Dictionary mapping cost function names to weights
            weight: Overall weight for this cost function
        """
        super().__init__(model, weight)
        self.name = "hybrid"
        
        # Store cost function specifications for later initialization
        self.cost_function_specs = cost_functions
        self.cost_functions = {}
        
        # We'll initialize the actual cost functions when needed to avoid circular imports
    
    def _init_cost_functions(self):
        """Initialize the cost functions if they haven't been initialized yet."""
        if not self.cost_functions:
            from . import get_cost_function
            for name, sub_weight in self.cost_function_specs.items():
                self.cost_functions[name] = get_cost_function(name, self.model, weight=sub_weight)
    
    def implement(self) -> moco.MocoGoal:
        """Implement the cost function as a Moco goal.
        
        Returns:
            Moco goal for hybrid cost
        """
        # Initialize cost functions if needed
        self._init_cost_functions()
        
        # Create a weighted sum goal
        hybrid = moco.MocoGoal("hybrid")
        
        # Add each sub-cost function to the problem
        for name, cost_func in self.cost_functions.items():
            sub_goal = cost_func.implement()
            hybrid.addGoal(sub_goal, cost_func.weight)
        
        # Set the overall weight
        hybrid.setWeight(self.weight)
        
        return hybrid
    
    def compute(self, states: Dict[str, np.ndarray], 
               controls: Dict[str, np.ndarray], 
               time: np.ndarray) -> float:
        """Compute the hybrid cost for given states and controls.
        
        Args:
            states: Dictionary of state trajectories
            controls: Dictionary of control trajectories
            time: Time points
            
        Returns:
            Total hybrid cost
        """
        # Initialize cost functions if needed
        self._init_cost_functions()
        
        total_cost = 0.0
        
        # Sum the weighted costs from each sub-cost function
        for name, cost_func in self.cost_functions.items():
            sub_cost = cost_func.compute(states, controls, time)
            total_cost += sub_cost
        
        # Apply overall weight
        total_cost *= self.weight
        
        return total_cost 