"""
Head motion cost function implementation.

This module implements a cost function that minimizes head acceleration
to maintain stable head motion during walking.
"""

import casadi as ca
import numpy as np
from opensim import Model, Body

class HeadMotion:
    """Head motion cost function that minimizes head acceleration."""
    
    def __init__(self, model: Model):
        """Initialize the cost function with the OpenSim model.
        
        Args:
            model: OpenSim model containing the head body
        """
        self.model = model
        self.head = model.getBodySet().get("head")
        
    def compute(self, states, controls, time):
        """Compute the head motion cost.
        
        Args:
            states: State variables (head position, velocity)
            controls: Control variables (not used)
            time: Time points
            
        Returns:
            CasADi expression for the head motion cost
        """
        # Get head acceleration components
        accel_x = states["head_accel_x"]
        accel_y = states["head_accel_y"]
        accel_z = states["head_accel_z"]
        
        # Compute squared acceleration magnitude
        accel_squared = accel_x**2 + accel_y**2 + accel_z**2
        
        # Integrate over time
        head_motion_integral = ca.integrator('head_motion_integral', 'cvodes',
                                           {'x': accel_squared},
                                           {'t0': time[0], 'tf': time[-1]})
        
        return head_motion_integral 