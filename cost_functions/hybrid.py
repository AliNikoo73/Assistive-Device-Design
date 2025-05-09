"""
Hybrid cost function implementation.

This module implements a cost function that combines cost of transport
and muscle effort with equal weighting.
"""

import casadi as ca
import numpy as np
from opensim import Model
from .cot import CostOfTransport
from .muscle_effort import MuscleEffort

class Hybrid:
    """Hybrid cost function combining CoT and muscle effort."""
    
    def __init__(self, model: Model):
        """Initialize the cost function with the OpenSim model.
        
        Args:
            model: OpenSim model containing the muscles
        """
        self.model = model
        self.cot = CostOfTransport(model)
        self.muscle_effort = MuscleEffort(model)
        
        # Equal weighting of CoT and muscle effort
        self.cot_weight = 0.5
        self.effort_weight = 0.5
        
    def compute(self, states, controls, time):
        """Compute the hybrid cost.
        
        Args:
            states: State variables (muscle activations, fiber lengths, etc.)
            controls: Control variables (muscle excitations)
            time: Time points
            
        Returns:
            CasADi expression for the hybrid cost
        """
        # Compute individual costs
        cot_cost = self.cot.compute(states, controls, time)
        effort_cost = self.muscle_effort.compute(states, controls, time)
        
        # Combine with weights
        hybrid_cost = (self.cot_weight * cot_cost + 
                      self.effort_weight * effort_cost)
        
        return hybrid_cost 