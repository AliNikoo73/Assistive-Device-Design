"""
Cost of transport (CoT) cost function implementation.

This module implements the metabolic cost of transport based on the
Umberger 2010 metabolic model.
"""

import casadi as ca
import numpy as np
from opensim import Model, Muscle

class CostOfTransport:
    """Cost of transport cost function based on Umberger 2010 metabolic model."""
    
    def __init__(self, model: Model):
        """Initialize the cost function with the OpenSim model.
        
        Args:
            model: OpenSim model containing the muscles
        """
        self.model = model
        self.muscles = [m for m in model.getMuscles()]
        
        # Metabolic parameters from Umberger 2010
        self.act_heat_rate = 40.0  # W/kg
        self.short_heat_rate = 133.0  # W/kg
        self.maint_heat_rate = 74.0  # W/kg
        self.work_rate = 0.0  # W/kg (will be computed)
        
    def compute(self, states, controls, time):
        """Compute the cost of transport.
        
        Args:
            states: State variables (muscle activations, fiber lengths, etc.)
            controls: Control variables (muscle excitations)
            time: Time points
            
        Returns:
            CasADi expression for the cost of transport
        """
        total_metabolic_power = 0.0
        
        for i, muscle in enumerate(self.muscles):
            # Get muscle states
            activation = states[f"{muscle.getName()}_activation"]
            fiber_length = states[f"{muscle.getName()}_fiber_length"]
            fiber_velocity = states[f"{muscle.getName()}_fiber_velocity"]
            
            # Get muscle control
            excitation = controls[f"{muscle.getName()}_excitation"]
            
            # Compute metabolic power components
            act_heat = self.act_heat_rate * activation
            short_heat = self.short_heat_rate * activation * fiber_velocity
            maint_heat = self.maint_heat_rate * activation
            
            # Compute mechanical work
            force = muscle.computeActuation(states)
            work = force * fiber_velocity
            
            # Total metabolic power for this muscle
            muscle_power = act_heat + short_heat + maint_heat + work
            
            # Add to total (weighted by muscle mass)
            total_metabolic_power += muscle_power * muscle.getMaxIsometricForce()
        
        # Compute cost of transport
        body_mass = self.model.getTotalMass()
        walking_speed = 1.25  # m/s (from paper)
        
        # Integrate power over time and divide by distance
        total_work = ca.integrator('total_work', 'cvodes',
                                 {'x': total_metabolic_power},
                                 {'t0': time[0], 'tf': time[-1]})
        
        distance = walking_speed * (time[-1] - time[0])
        cot = total_work / (body_mass * distance)
        
        return cot 