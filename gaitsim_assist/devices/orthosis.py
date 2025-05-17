"""
Orthosis device module.

This module provides classes for modeling orthotic devices in gait simulations.
"""

import numpy as np
import opensim as osim
from typing import Dict, List, Optional, Union, Any

from .base import AssistiveDevice


class Orthosis(AssistiveDevice):
    """Class for modeling orthotic devices."""
    
    def __init__(self, name: str, model: osim.Model, joint_name: str, 
                mass: float = 0.5, stiffness: float = 50.0, 
                damping: float = 2.0, range_min: float = -1.0, range_max: float = 1.0):
        """Initialize the orthotic device.
        
        Args:
            name: Name of the orthotic device
            model: OpenSim model to which the device will be added
            joint_name: Name of the joint to modify
            mass: Mass of the orthotic device in kg
            stiffness: Stiffness of the orthotic device in Nm/rad
            damping: Damping coefficient of the orthotic device in Nms/rad
            range_min: Minimum allowed joint angle in radians
            range_max: Maximum allowed joint angle in radians
        """
        super().__init__(name, model)
        self.joint_name = joint_name
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping
        self.range_min = range_min
        self.range_max = range_max
        
        # Find the joint to modify
        self.joint = None
        for i in range(self.model.getJointSet().getSize()):
            joint = self.model.getJointSet().get(i)
            if joint.getName() == self.joint_name:
                self.joint = joint
                break
        
        if self.joint is None:
            raise ValueError(f"Joint '{self.joint_name}' not found in the model")
    
    def apply_to_model(self):
        """Apply the orthotic device to the model."""
        # Get the coordinate for the joint
        if self.joint.numCoordinates() == 0:
            raise ValueError(f"Joint '{self.joint_name}' has no coordinates")
        
        coord = self.joint.get_coordinates(0)
        coord_name = coord.getName()
        
        # Add mass to the child body to represent the orthosis
        child_body = self.model.getBodySet().get(
            self.joint.getChildFrame().getParentFrame().getName())
        original_mass = child_body.getMass()
        child_body.setMass(original_mass + self.mass)
        
        # Add a spring force to represent the orthosis stiffness
        spring = osim.SpringGeneralizedForce(coord_name)
        spring.setName(f"{self.name}_spring")
        spring.setStiffness(self.stiffness)
        spring.setRestLength(0.0)  # Neutral position
        self.model.addForce(spring)
        
        # Add a damper force to represent the orthosis damping
        damper = osim.DampingGeneralizedForce(coord_name)
        damper.setName(f"{self.name}_damper")
        damper.setDamping(self.damping)
        self.model.addForce(damper)
        
        # Add coordinate limit constraints
        if self.range_min is not None:
            min_constraint = osim.CoordinateLimitForce(coord_name, self.range_min, 1e3, 1.0, 5.0)
            min_constraint.setName(f"{self.name}_min_limit")
            self.model.addForce(min_constraint)
        
        if self.range_max is not None:
            max_constraint = osim.CoordinateLimitForce(coord_name, self.range_max, 1e3, 1.0, 5.0, True)
            max_constraint.setName(f"{self.name}_max_limit")
            self.model.addForce(max_constraint)
    
    def get_parameters(self) -> Dict[str, float]:
        """Get the parameters of the orthotic device.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'mass': self.mass,
            'stiffness': self.stiffness,
            'damping': self.damping,
            'range_min': self.range_min,
            'range_max': self.range_max
        }
    
    def set_parameters(self, parameters: Dict[str, float]):
        """Set the parameters of the orthotic device.
        
        Args:
            parameters: Dictionary of parameters
        """
        if 'mass' in parameters:
            self.mass = parameters['mass']
        
        if 'stiffness' in parameters:
            self.stiffness = parameters['stiffness']
        
        if 'damping' in parameters:
            self.damping = parameters['damping']
        
        if 'range_min' in parameters:
            self.range_min = parameters['range_min']
        
        if 'range_max' in parameters:
            self.range_max = parameters['range_max']
    
    def get_mass(self) -> float:
        """Get the mass of the orthotic device.
        
        Returns:
            Mass in kg
        """
        return self.mass 