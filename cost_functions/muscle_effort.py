"""
Muscle effort cost function implementation.

This module implements a cost function that minimizes the sum of squared
muscle excitations over time.
"""

import casadi as ca
import numpy as np
from opensim import Model

class MuscleEffort:
    """Muscle effort cost function that minimizes squared excitations."""
    
    def __init__(self, model: Model):
        """Initialize the cost function with the OpenSim model.
        
        Args:
            model: OpenSim model containing the muscles
        """
        self.model = model
        self.muscles = [m for m in model.getMuscles()]
        
    def compute(self, states, controls, time):
        """Compute the muscle effort cost.
        
        Args:
            states: State variables (muscle activations, fiber lengths, etc.)
            controls: Control variables (muscle excitations)
            time: Time points
            
        Returns:
            CasADi expression for the muscle effort cost
        """
        total_effort = 0.0
        
        for muscle in self.muscles:
            # Get muscle control (excitation)
            excitation = controls[f"{muscle.getName()}_excitation"]
            
            # Add squared excitation to total effort
            total_effort += excitation**2
        
        # Integrate over time
        effort_integral = ca.integrator('effort_integral', 'cvodes',
                                      {'x': total_effort},
                                      {'t0': time[0], 'tf': time[-1]})
        
        return effort_integral 