"""
Unit tests for cost function gradients.

This module verifies that the cost function gradients computed using
CasADi automatic differentiation match finite difference approximations.
"""

import unittest
import numpy as np
from opensim import Model
import moco
from cost_functions import (
    CostOfTransport,
    MuscleEffort,
    JointTorque,
    Fatigue,
    HeadMotion,
    Hybrid
)

class TestCostFunctionGradients(unittest.TestCase):
    """Test case for cost function gradients."""
    
    def setUp(self):
        """Set up test case."""
        # Load model
        self.model = Model('../models/planar18.osim')
        
        # Create test problem
        self.problem = moco.MocoProblem()
        self.problem.setModel(self.model)
        self.problem.setTimeBounds(0.0, 1.0)
        
        # Create solver
        self.solver = moco.MocoCasADiSolver()
        self.solver.setProblem(self.problem)
        self.solver.set_num_mesh_points(2)  # Minimal problem
        
        # Create test trajectory
        self.traj = moco.MocoTrajectory()
        self.traj.setNumTimes(2)
        self.traj.setTime([0.0, 1.0])
        
        # Set random states and controls
        for state in self.problem.getStateNames():
            self.traj.setState(state, np.random.rand(2))
        for control in self.problem.getControlNames():
            self.traj.setControl(control, np.random.rand(2))
    
    def test_cot_gradients(self):
        """Test cost of transport gradients."""
        # Set cost function
        self.problem.setCostFunction(CostOfTransport(self.model))
        
        # Get CasADi gradients
        casadi_grad = self.solver.getCostGradient(self.traj)
        
        # Get finite difference gradients
        fd_grad = self.solver.getCostGradientFiniteDifference(self.traj)
        
        # Compare
        np.testing.assert_allclose(casadi_grad, fd_grad, rtol=1e-5)
    
    def test_muscle_effort_gradients(self):
        """Test muscle effort gradients."""
        # Set cost function
        self.problem.setCostFunction(MuscleEffort(self.model))
        
        # Get CasADi gradients
        casadi_grad = self.solver.getCostGradient(self.traj)
        
        # Get finite difference gradients
        fd_grad = self.solver.getCostGradientFiniteDifference(self.traj)
        
        # Compare
        np.testing.assert_allclose(casadi_grad, fd_grad, rtol=1e-5)
    
    def test_joint_torque_gradients(self):
        """Test joint torque gradients."""
        # Set cost function
        self.problem.setCostFunction(JointTorque(self.model))
        
        # Get CasADi gradients
        casadi_grad = self.solver.getCostGradient(self.traj)
        
        # Get finite difference gradients
        fd_grad = self.solver.getCostGradientFiniteDifference(self.traj)
        
        # Compare
        np.testing.assert_allclose(casadi_grad, fd_grad, rtol=1e-5)
    
    def test_fatigue_gradients(self):
        """Test fatigue gradients."""
        # Set cost function
        self.problem.setCostFunction(Fatigue(self.model))
        
        # Get CasADi gradients
        casadi_grad = self.solver.getCostGradient(self.traj)
        
        # Get finite difference gradients
        fd_grad = self.solver.getCostGradientFiniteDifference(self.traj)
        
        # Compare
        np.testing.assert_allclose(casadi_grad, fd_grad, rtol=1e-5)
    
    def test_head_motion_gradients(self):
        """Test head motion gradients."""
        # Set cost function
        self.problem.setCostFunction(HeadMotion(self.model))
        
        # Get CasADi gradients
        casadi_grad = self.solver.getCostGradient(self.traj)
        
        # Get finite difference gradients
        fd_grad = self.solver.getCostGradientFiniteDifference(self.traj)
        
        # Compare
        np.testing.assert_allclose(casadi_grad, fd_grad, rtol=1e-5)
    
    def test_hybrid_gradients(self):
        """Test hybrid cost function gradients."""
        # Set cost function
        self.problem.setCostFunction(Hybrid(self.model))
        
        # Get CasADi gradients
        casadi_grad = self.solver.getCostGradient(self.traj)
        
        # Get finite difference gradients
        fd_grad = self.solver.getCostGradientFiniteDifference(self.traj)
        
        # Compare
        np.testing.assert_allclose(casadi_grad, fd_grad, rtol=1e-5)

if __name__ == '__main__':
    unittest.main() 