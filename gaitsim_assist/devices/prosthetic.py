"""
Prosthetic device module.

This module provides classes for modeling prosthetic devices in gait simulations.
"""

import numpy as np
import opensim as osim
from typing import Dict, List, Optional, Union, Any

from .base import AssistiveDevice


class Prosthetic(AssistiveDevice):
    """Class for modeling prosthetic devices."""
    
    def __init__(self, name: str, model: osim.Model, limb_name: str, 
                mass: float = 1.0, stiffness: float = 100.0, 
                damping: float = 5.0):
        """Initialize the prosthetic device.
        
        Args:
            name: Name of the prosthetic device
            model: OpenSim model to which the device will be added
            limb_name: Name of the limb to replace (e.g., 'foot', 'shank')
            mass: Mass of the prosthetic device in kg
            stiffness: Stiffness of the prosthetic device in N/m or Nm/rad
            damping: Damping coefficient of the prosthetic device in Ns/m or Nms/rad
        """
        super().__init__(name, model)
        self.limb_name = limb_name
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping
        
        # Store original model components for later restoration
        self._store_original_components()
    
    def _store_original_components(self):
        """Store original model components for later restoration."""
        self.original_bodies = []
        self.original_joints = []
        self.original_muscles = []
        
        # Find the body to replace
        target_body = None
        for i in range(self.model.getBodySet().getSize()):
            body = self.model.getBodySet().get(i)
            if body.getName() == self.limb_name:
                target_body = body
                self.original_bodies.append(body)
                break
        
        if target_body is None:
            raise ValueError(f"Body '{self.limb_name}' not found in the model")
        
        # Find joints connected to the target body
        for i in range(self.model.getJointSet().getSize()):
            joint = self.model.getJointSet().get(i)
            if (joint.getParentFrame().getParentFrame().getName() == self.limb_name or
                joint.getChildFrame().getParentFrame().getName() == self.limb_name):
                self.original_joints.append(joint)
        
        # Find muscles attached to the target body
        for i in range(self.model.getMuscles().getSize()):
            muscle = self.model.getMuscles().get(i)
            # Check if any attachment point is on the target body
            for j in range(muscle.getGeometryPath().getPathPointSet().getSize()):
                point = muscle.getGeometryPath().getPathPointSet().get(j)
                if point.getParentFrame().getName() == self.limb_name:
                    self.original_muscles.append(muscle)
                    break
    
    def apply_to_model(self):
        """Apply the prosthetic device to the model."""
        # Create a new prosthetic body
        prosthetic_body = osim.Body(f"{self.name}_{self.limb_name}", 
                                  self.mass, 
                                  osim.Vec3(0),  # Center of mass
                                  osim.Inertia(0.1, 0.1, 0.1))  # Inertia tensor
        self.model.addBody(prosthetic_body)
        
        # Find the parent body of the original limb
        parent_body = None
        parent_joint = None
        for joint in self.original_joints:
            if joint.getChildFrame().getParentFrame().getName() == self.limb_name:
                parent_joint = joint
                parent_body = self.model.getBodySet().get(
                    joint.getParentFrame().getParentFrame().getName())
                break
        
        if parent_body is None:
            raise ValueError(f"Could not find parent body for '{self.limb_name}'")
        
        # Create a new joint connecting the parent body to the prosthetic
        if isinstance(parent_joint, osim.PinJoint):
            # Create a pin joint with spring and damper
            new_joint = osim.PinJoint(f"{self.name}_joint", 
                                    parent_body, parent_joint.getParentFrame().get_translation(), 
                                    parent_joint.getParentFrame().get_orientation(),
                                    prosthetic_body, osim.Vec3(0), osim.Vec3(0))
            
            # Add a spring and damper to the joint
            coord = new_joint.updCoordinate()
            coord.setName(f"{self.name}_angle")
            
            # Add spring force
            spring = osim.SpringGeneralizedForce(coord.getName())
            spring.setStiffness(self.stiffness)
            spring.setRestLength(0.0)
            self.model.addForce(spring)
            
            # Add damper force
            damper = osim.DampingGeneralizedForce(coord.getName())
            damper.setDamping(self.damping)
            self.model.addForce(damper)
        else:
            # For other joint types, create a custom joint
            new_joint = osim.CustomJoint(f"{self.name}_joint",
                                       parent_body, parent_joint.getParentFrame().get_translation(),
                                       parent_joint.getParentFrame().get_orientation(),
                                       prosthetic_body, osim.Vec3(0), osim.Vec3(0),
                                       osim.SpatialTransform())
            
            # Add spatial transform functions
            transform = new_joint.updSpatialTransform()
            
            # Add translations
            for i in range(3):
                transform.updTranslation(i).setFunction(osim.Constant(0))
            
            # Add rotations with spring-like behavior
            for i in range(3):
                transform.updRotation(i).setFunction(osim.Constant(0))
        
        # Add the joint to the model
        self.model.addJoint(new_joint)
        
        # Disable the original limb and its connections
        self._disable_original_components()
    
    def _disable_original_components(self):
        """Disable original model components."""
        # This is a simplified implementation
        # In a real implementation, we would need to carefully handle
        # the removal or disabling of original components
        
        # For now, we just scale the mass of the original body to near zero
        for body in self.original_bodies:
            body.setMass(1e-6)
    
    def get_parameters(self) -> Dict[str, float]:
        """Get the parameters of the prosthetic device.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'mass': self.mass,
            'stiffness': self.stiffness,
            'damping': self.damping
        }
    
    def set_parameters(self, parameters: Dict[str, float]):
        """Set the parameters of the prosthetic device.
        
        Args:
            parameters: Dictionary of parameters
        """
        if 'mass' in parameters:
            self.mass = parameters['mass']
        
        if 'stiffness' in parameters:
            self.stiffness = parameters['stiffness']
        
        if 'damping' in parameters:
            self.damping = parameters['damping']
    
    def get_mass(self) -> float:
        """Get the mass of the prosthetic device.
        
        Returns:
            Mass in kg
        """
        return self.mass 