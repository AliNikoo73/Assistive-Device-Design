"""
Assistive device models for GaitSim Assist.

This module provides models for various assistive devices that can be
incorporated into gait simulations, including:
- Exoskeletons: Devices that provide external torque to joints
- Prosthetics: Replacement limbs with customizable properties
- Orthoses: Devices that modify joint stiffness or range of motion
"""

from .base import AssistiveDevice
from .exoskeleton import Exoskeleton
from .prosthetic import Prosthetic
from .orthosis import Orthosis

__all__ = [
    'AssistiveDevice',
    'Exoskeleton',
    'Prosthetic',
    'Orthosis'
] 