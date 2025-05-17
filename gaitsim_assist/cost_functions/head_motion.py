"""
Head motion cost function for GaitSim Assist.

This module provides a cost function that minimizes head motion,
defined as the acceleration of the head center of mass.
"""

import numpy as np
import opensim as osim
import opensim.moco as moco
from typing import Dict, List, Optional, Union, Any

from .base import CostFunction


class HeadMotion(CostFunction):
    """Cost function that minimizes head motion."""
    
    def __init__(self, model: osim.Model, weight: float = 1.0):
        """Initialize the head motion cost function.
        
        Args:
            model: OpenSim model
            weight: Weight for this cost function
        """
        super().__init__(model, weight)
        self.name = "head_motion"
        
        # Find the head body in the model
        self.head_body = None
        for i in range(model.getBodySet().getSize()):
            body = model.getBodySet().get(i)
            if "head" in body.getName().lower():
                self.head_body = body.getName()
                break
        
        if self.head_body is None:
            # If no head body is found, use the most superior body
            most_superior_height = -float('inf')
            for i in range(model.getBodySet().getSize()):
                body = model.getBodySet().get(i)
                if body.getName() != "ground":
                    height = body.getPositionInGround(model.getWorkingState())[1]
                    if height > most_superior_height:
                        most_superior_height = height
                        self.head_body = body.getName()
    
    def implement(self) -> moco.MocoGoal:
        """Implement the cost function as a Moco goal.
        
        Returns:
            Moco goal for head motion
        """
        # Create an acceleration tracking goal
        head_motion = moco.MocoAccelerationTrackingGoal("head_motion")
        
        # Add the head body to track
        if self.head_body:
            head_motion.addAccelerationFrame(self.model.getBodySet().get(self.head_body))
            
            # Set reference acceleration to zero (minimize acceleration)
            head_motion.setAccelerationReference(moco.StdVectorVec3(
                [osim.Vec3(0.0, 0.0, 0.0)] * 101))  # 101 points for the full trajectory
        
        # Set the weight
        head_motion.setWeight(self.weight)
        
        return head_motion
    
    def compute(self, states: Dict[str, np.ndarray], 
               controls: Dict[str, np.ndarray], 
               time: np.ndarray) -> float:
        """Compute the head motion cost for given states and controls.
        
        Args:
            states: Dictionary of state trajectories
            controls: Dictionary of control trajectories
            time: Time points
            
        Returns:
            Total head motion cost
        """
        # This is a simplified calculation that approximates head acceleration
        # from position states. In a real implementation, we would compute
        # actual accelerations using OpenSim's dynamics engine.
        
        if not self.head_body:
            return 0.0
            
        total_cost = 0.0
        
        # Look for head position states
        head_pos_x = None
        head_pos_y = None
        
        for state_name in states:
            if self.head_body in state_name and "/position" in state_name:
                if "/x" in state_name:
                    head_pos_x = states[state_name]
                elif "/y" in state_name:
                    head_pos_y = states[state_name]
        
        # If we have position data, compute approximate accelerations
        if head_pos_x is not None and head_pos_y is not None:
            # Compute time steps
            dt = np.diff(time)
            
            # Compute velocities (first derivative)
            vel_x = np.diff(head_pos_x) / dt
            vel_y = np.diff(head_pos_y) / dt
            
            # Compute accelerations (second derivative)
            acc_x = np.diff(vel_x) / dt[:-1]
            acc_y = np.diff(vel_y) / dt[:-1]
            
            # Sum squared accelerations
            total_cost = np.sum(acc_x**2 + acc_y**2)
        
        # Apply weight
        total_cost *= self.weight
        
        return total_cost 