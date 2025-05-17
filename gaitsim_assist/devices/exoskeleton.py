"""
Exoskeleton device model for GaitSim Assist.

This module implements a configurable exoskeleton device that can provide
assistive torques to various joints.
"""

import numpy as np
import opensim as osim
from typing import Dict, Any, List, Optional, Union, Tuple

from .base import AssistiveDevice


class Exoskeleton(AssistiveDevice):
    """Configurable exoskeleton device model."""
    
    def __init__(self, name: str, model: osim.Model, 
                joint_name: str = 'ankle',
                mass: float = 1.0,
                max_torque: float = 100.0,
                stiffness: float = 0.0,
                damping: float = 0.0,
                control_mode: str = 'torque'):
        """Initialize the exoskeleton device.
        
        Args:
            name: Name of the device
            model: OpenSim model to which the device will be added
            joint_name: Name of the joint to assist (ankle, knee, hip)
            mass: Mass of the device in kg
            max_torque: Maximum torque the device can provide in Nm
            stiffness: Stiffness coefficient in Nm/rad
            damping: Damping coefficient in Nm/(rad/s)
            control_mode: Control mode ('torque', 'position', 'impedance')
        """
        super().__init__(name, model)
        self.joint_name = joint_name
        self.mass = mass
        self.max_torque = max_torque
        self.stiffness = stiffness
        self.damping = damping
        self.control_mode = control_mode
        
        # Validate parameters
        self._validate_parameters()
        
        # Find the target joint in the model
        self.target_joint = self._find_target_joint()
        
    def _validate_parameters(self) -> None:
        """Validate the device parameters."""
        if self.mass <= 0:
            raise ValueError(f"Mass must be positive, got {self.mass}")
        
        if self.max_torque <= 0:
            raise ValueError(f"Maximum torque must be positive, got {self.max_torque}")
        
        if self.stiffness < 0:
            raise ValueError(f"Stiffness must be non-negative, got {self.stiffness}")
        
        if self.damping < 0:
            raise ValueError(f"Damping must be non-negative, got {self.damping}")
        
        valid_control_modes = ['torque', 'position', 'impedance']
        if self.control_mode not in valid_control_modes:
            raise ValueError(f"Control mode must be one of {valid_control_modes}, "
                           f"got {self.control_mode}")
        
        valid_joints = ['ankle', 'knee', 'hip']
        if self.joint_name not in valid_joints:
            raise ValueError(f"Joint name must be one of {valid_joints}, "
                           f"got {self.joint_name}")
    
    def _find_target_joint(self) -> osim.Joint:
        """Find the target joint in the model.
        
        Returns:
            OpenSim Joint object
        """
        # Map joint names to actual joint names in the model
        joint_map = {
            'ankle': 'ankle',
            'knee': 'knee',
            'hip': 'hip'
        }
        
        # Get the actual joint name
        actual_joint_name = joint_map[self.joint_name]
        
        # Find the joint in the model
        joint = self.model.getJointSet().get(actual_joint_name)
        if joint is None:
            raise ValueError(f"Joint '{actual_joint_name}' not found in the model")
        
        return joint
    
    def apply_to_model(self) -> None:
        """Apply the exoskeleton to the model."""
        # Add a coordinate actuator to the target joint
        actuator = osim.CoordinateActuator(self.target_joint.getCoordinate().getName())
        actuator.setName(f"{self.name}_actuator")
        actuator.setOptimalForce(self.max_torque)
        actuator.setMinControl(-1.0)
        actuator.setMaxControl(1.0)
        
        # Add the actuator to the model
        self.model.addForce(actuator)
        self.components.append(actuator)
        
        # Add a controller based on the control mode
        if self.control_mode == 'torque':
            controller = self._create_torque_controller()
        elif self.control_mode == 'position':
            controller = self._create_position_controller()
        elif self.control_mode == 'impedance':
            controller = self._create_impedance_controller()
        
        # Add the controller to the model
        self.model.addController(controller)
        self.components.append(controller)
        
        # Add a body to represent the exoskeleton mass
        self._add_exo_body()
    
    def _create_torque_controller(self) -> osim.Controller:
        """Create a torque controller.
        
        Returns:
            OpenSim Controller object
        """
        # Create a PrescribedController
        controller = osim.PrescribedController()
        controller.setName(f"{self.name}_controller")
        
        # Add the actuator to the controller
        controller.addActuator(self.model.getForceSet().get(f"{self.name}_actuator"))
        
        # Create a torque function based on gait phase
        # This is a simple example that provides assistance during push-off
        torque_function = osim.PiecewiseLinearFunction()
        torque_function.addPoint(0.0, 0.0)  # No torque at heel strike
        torque_function.addPoint(0.4, 0.0)  # No torque in mid-stance
        torque_function.addPoint(0.6, 1.0)  # Max torque at push-off
        torque_function.addPoint(0.8, 0.0)  # No torque in swing
        torque_function.addPoint(1.0, 0.0)  # No torque at heel strike
        
        # Set the function for the controller
        controller.prescribeControlForActuator(0, torque_function)
        
        return controller
    
    def _create_position_controller(self) -> osim.Controller:
        """Create a position controller.
        
        Returns:
            OpenSim Controller object
        """
        # Create a PID controller
        controller = osim.PIDController(f"{self.name}_controller")
        
        # Add the actuator to the controller
        controller.addActuator(self.model.getForceSet().get(f"{self.name}_actuator"))
        
        # Add the control sensor (joint angle)
        sensor = osim.CoordinateSensor(self.target_joint.getCoordinate().getName())
        controller.addSensor(sensor)
        
        # Set PID gains
        controller.setKp(100.0)  # Proportional gain
        controller.setKi(0.0)    # Integral gain
        controller.setKd(10.0)   # Derivative gain
        
        # Set the reference value (desired joint angle)
        reference_function = osim.Constant(0.0)  # Neutral position
        controller.setReference(reference_function)
        
        return controller
    
    def _create_impedance_controller(self) -> osim.Controller:
        """Create an impedance controller.
        
        Returns:
            OpenSim Controller object
        """
        # For impedance control, we'll use a custom controller
        # that implements a spring-damper system
        controller = osim.PrescribedController()
        controller.setName(f"{self.name}_controller")
        
        # Add the actuator to the controller
        controller.addActuator(self.model.getForceSet().get(f"{self.name}_actuator"))
        
        # We'll use a linear function of joint angle and velocity
        # τ = -k(θ - θ₀) - bθ̇
        # This is implemented in the model's computeControls method
        
        return controller
    
    def _add_exo_body(self) -> None:
        """Add a body to represent the exoskeleton mass."""
        # Get the parent body of the target joint
        parent_body = self.target_joint.getParentBody()
        
        # Create a new body for the exoskeleton
        exo_body = osim.Body(f"{self.name}_body", self.mass, osim.Vec3(0),
                           osim.Inertia(0.01, 0.01, 0.01))
        
        # Add the body to the model
        self.model.addBody(exo_body)
        self.components.append(exo_body)
        
        # Create a weld joint to attach the exoskeleton to the parent body
        weld_joint = osim.WeldJoint(f"{self.name}_joint", 
                                  parent_body, osim.Vec3(0), osim.Vec3(0),
                                  exo_body, osim.Vec3(0), osim.Vec3(0))
        
        # Add the joint to the model
        self.model.addJoint(weld_joint)
        self.components.append(weld_joint)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the device parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
        return {
            'joint_name': self.joint_name,
            'mass': self.mass,
            'max_torque': self.max_torque,
            'stiffness': self.stiffness,
            'damping': self.damping,
            'control_mode': self.control_mode
        }
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set the device parameters.
        
        Args:
            parameters: Dictionary of parameter names and values
        """
        # Update parameters
        if 'joint_name' in parameters:
            self.joint_name = parameters['joint_name']
        
        if 'mass' in parameters:
            self.mass = parameters['mass']
        
        if 'max_torque' in parameters:
            self.max_torque = parameters['max_torque']
        
        if 'stiffness' in parameters:
            self.stiffness = parameters['stiffness']
        
        if 'damping' in parameters:
            self.damping = parameters['damping']
        
        if 'control_mode' in parameters:
            self.control_mode = parameters['control_mode']
        
        # Validate the updated parameters
        self._validate_parameters()
        
        # If the device has already been applied to the model,
        # remove it and reapply with the new parameters
        if self.components:
            self.remove_from_model()
            self.target_joint = self._find_target_joint()
            self.apply_to_model() 