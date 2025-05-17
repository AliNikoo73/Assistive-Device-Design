"""
Base assistive device class for GaitSim Assist.

This module defines the base AssistiveDevice class that all assistive device models
must inherit from.
"""

import numpy as np
import opensim as osim
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union


class AssistiveDevice(ABC):
    """Base class for all assistive devices."""
    
    def __init__(self, name: str, model: osim.Model):
        """Initialize the assistive device.
        
        Args:
            name: Name of the device
            model: OpenSim model to which the device will be added
        """
        self.name = name
        self.model = model
        self.components = []
        
    @abstractmethod
    def apply_to_model(self) -> None:
        """Apply the assistive device to the OpenSim model.
        
        This method should add the necessary components (bodies, joints, forces, etc.)
        to the model to represent the assistive device.
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get the device parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set the device parameters.
        
        Args:
            parameters: Dictionary of parameter names and values
        """
        pass
    
    def get_mass(self) -> float:
        """Get the total mass of the device.
        
        Returns:
            Total mass in kg
        """
        return sum(component.getMass() for component in self.components 
                  if hasattr(component, 'getMass'))
    
    def get_forces(self) -> List[osim.Force]:
        """Get all forces applied by the device.
        
        Returns:
            List of OpenSim Force objects
        """
        return [component for component in self.components 
               if isinstance(component, osim.Force)]
    
    def remove_from_model(self) -> None:
        """Remove the assistive device from the model.
        
        This method removes all components added by the device from the model.
        """
        for component in reversed(self.components):
            if isinstance(component, osim.Body):
                self.model.removeBody(component)
            elif isinstance(component, osim.Joint):
                self.model.removeJoint(component)
            elif isinstance(component, osim.Force):
                self.model.removeForce(component)
            elif isinstance(component, osim.Controller):
                self.model.removeController(component)
    
    def __str__(self) -> str:
        """Get a string representation of the device.
        
        Returns:
            String representation
        """
        params = self.get_parameters()
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}(name='{self.name}', {params_str})" 