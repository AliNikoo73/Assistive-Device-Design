"""
Cost of transport (CoT) cost function implementation.

This module implements the metabolic cost of transport based on the
Umberger 2010 metabolic model.
"""

import numpy as np
import opensim as osim
import opensim.moco as moco
from typing import Dict, Any, Union

from .base import CostFunction


class CostOfTransport(CostFunction):
    """Cost of transport cost function based on Umberger 2010 metabolic model."""
    
    def __init__(self, model, weight: float = 1.0, name: str = None):
        """Initialize the cost function with the OpenSim model.
        
        Args:
            model: OpenSim model containing the muscles
            weight: Weight of this cost function when combined with others
            name: Optional name for this cost function instance
        """
        super().__init__(model, weight, name)
        
        # Get list of muscles in the model
        self.muscles = []
        for i in range(model.getMuscles().getSize()):
            self.muscles.append(model.getMuscles().get(i))
        
        # Metabolic parameters from Umberger 2010
        self.act_heat_rate = 40.0  # W/kg
        self.short_heat_rate = 133.0  # W/kg
        self.maint_heat_rate = 74.0  # W/kg
        self.work_rate = 0.0  # W/kg (will be computed)
        
    def implement(self) -> moco.MocoGoal:
        """Implement the cost function as a Moco goal.
        
        Returns:
            Moco goal for cost of transport
        """
        # Create a metabolic cost goal using Umberger2010 model
        cot = moco.MocoMetabolicCost("cost_of_transport")
        
        # Set to use Umberger2010 model
        cot.setMetabolicModel("Umberger2010")
        
        # Add all muscles to the cost
        for muscle in self.muscles:
            cot.addMuscle(muscle.getName())
        
        # Set the weight
        cot.setWeight(self.weight)
        
        # Set to compute CoT by dividing by distance
        cot.setDivideByDisplacement(True)
        
        return cot
        
    def compute(self, states: Dict[str, np.ndarray], controls: Dict[str, np.ndarray], 
               time: np.ndarray) -> float:
        """Compute the cost of transport.
        
        Args:
            states: State variables (muscle activations, fiber lengths, etc.)
            controls: Control variables (muscle excitations)
            time: Time points
            
        Returns:
            Cost of transport value
        """
        total_metabolic_power = 0.0
        
        for muscle in self.muscles:
            muscle_name = muscle.getName()
            
            # Get muscle states
            activation_key = None
            fiber_length_key = None
            fiber_velocity_key = None
            
            # Find the correct keys for this muscle's states
            for key in states:
                if muscle_name in key and "activation" in key:
                    activation_key = key
                elif muscle_name in key and "fiber_length" in key:
                    fiber_length_key = key
                elif muscle_name in key and "fiber_velocity" in key:
                    fiber_velocity_key = key
            
            # Get muscle states if available
            activation = states.get(activation_key, np.zeros_like(time))
            fiber_length = states.get(fiber_length_key, np.zeros_like(time))
            fiber_velocity = states.get(fiber_velocity_key, np.zeros_like(time))
            
            # Compute metabolic power components
            act_heat = self.act_heat_rate * activation
            short_heat = self.short_heat_rate * activation * fiber_velocity
            maint_heat = self.maint_heat_rate * activation
            
            # Compute mechanical work (simplified)
            force = muscle.getMaxIsometricForce() * activation
            work = force * fiber_velocity
            
            # Total metabolic power for this muscle
            muscle_power = act_heat + short_heat + maint_heat + work
            
            # Add to total (weighted by muscle mass)
            total_metabolic_power += np.sum(muscle_power * muscle.getMaxIsometricForce())
        
        # Compute cost of transport
        body_mass = self.model.getTotalMass()
        walking_speed = 1.25  # m/s (typical walking speed)
        
        # Integrate power over time and divide by distance
        dt = time[1] - time[0] if len(time) > 1 else 1.0
        total_work = np.sum(total_metabolic_power) * dt
        
        distance = walking_speed * (time[-1] - time[0])
        cot = total_work / (body_mass * distance)
        
        return cot 