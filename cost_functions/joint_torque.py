"""
Joint torque cost function implementation.

This module implements a cost function that minimizes the sum of squared
joint torques over time.
"""

import casadi as ca
import numpy as np
from opensim import Model, Coordinate

class JointTorque:
    """Joint torque cost function that minimizes squared joint torques."""
    
    def __init__(self, model: Model):
        """Initialize the cost function with the OpenSim model.
        
        Args:
            model: OpenSim model containing the joints
        """
        self.model = model
        self.coordinates = [c for c in model.getCoordinates()]
        
    def compute(self, states, controls, time):
        """Compute the joint torque cost.
        
        Args:
            states: State variables (joint angles, velocities)
            controls: Control variables (joint torques)
            time: Time points
            
        Returns:
            CasADi expression for the joint torque cost
        """
        total_torque = 0.0
        
        for coord in self.coordinates:
            # Get joint torque control
            torque = controls[f"{coord.getName()}_torque"]
            
            # Add squared torque to total
            total_torque += torque**2
        
        # Integrate over time
        torque_integral = ca.integrator('torque_integral', 'cvodes',
                                      {'x': total_torque},
                                      {'t0': time[0], 'tf': time[-1]})
        
        return torque_integral 