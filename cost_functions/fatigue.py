"""
Fatigue cost function implementation.

This module implements a cost function that minimizes muscle fatigue by
penalizing high muscle activations over time.
"""

import casadi as ca
import numpy as np
from opensim import Model

class Fatigue:
    """Fatigue cost function that minimizes muscle fatigue."""
    
    def __init__(self, model: Model):
        """Initialize the cost function with the OpenSim model.
        
        Args:
            model: OpenSim model containing the muscles
        """
        self.model = model
        self.muscles = [m for m in model.getMuscles()]
        
    def compute(self, states, controls, time):
        """Compute the fatigue cost.
        
        Args:
            states: State variables (muscle activations)
            controls: Control variables (muscle excitations)
            time: Time points
            
        Returns:
            CasADi expression for the fatigue cost
        """
        total_fatigue = 0.0
        
        for muscle in self.muscles:
            # Get muscle activation
            activation = states[f"{muscle.getName()}_activation"]
            
            # Add cubed activation to total fatigue
            # Using cube to penalize high activations more strongly
            total_fatigue += activation**3
        
        # Integrate over time
        fatigue_integral = ca.integrator('fatigue_integral', 'cvodes',
                                       {'x': total_fatigue},
                                       {'t0': time[0], 'tf': time[-1]})
        
        return fatigue_integral 