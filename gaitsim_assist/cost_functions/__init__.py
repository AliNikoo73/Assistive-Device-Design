"""
Cost function implementations for assistive device design optimization.

This module provides various cost functions for optimizing gait simulations:
- Cost of transport (CoT): Metabolic energy expenditure per distance traveled
- Muscle effort: Sum of squared muscle activations
- Joint torque: Sum of squared joint torques
- Fatigue: Maximum muscle activation
- Head motion: Minimization of head acceleration
- Hybrid: Combination of multiple cost functions with weights
"""

from .base import CostFunction
from .cot import CostOfTransport
from .muscle_effort import MuscleEffort
from .joint_torque import JointTorque
from .fatigue import Fatigue
from .head_motion import HeadMotion
from .hybrid import Hybrid

__all__ = [
    'CostFunction',
    'CostOfTransport',
    'MuscleEffort',
    'JointTorque',
    'Fatigue',
    'HeadMotion',
    'Hybrid',
    'get_cost_function'
]

def get_cost_function(name: str, model, **kwargs):
    """Get a cost function by name.
    
    Args:
        name: Name of the cost function
        model: OpenSim model
        **kwargs: Additional arguments to pass to the cost function constructor
        
    Returns:
        Cost function instance
    """
    cost_functions = {
        'cot': CostOfTransport,
        'cost_of_transport': CostOfTransport,
        'muscle_effort': MuscleEffort,
        'joint_torque': JointTorque,
        'fatigue': Fatigue,
        'head_motion': HeadMotion,
        'hybrid': Hybrid
    }
    
    name = name.lower()
    if name not in cost_functions:
        raise ValueError(f"Unknown cost function: {name}")
    
    return cost_functions[name](model, **kwargs) 