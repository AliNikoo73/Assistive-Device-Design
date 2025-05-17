"""
Basic tests for the GaitSim Assist library.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gaitsim_assist as gsa
from gaitsim_assist.cost_functions import CostOfTransport, MuscleEffort, Hybrid
from gaitsim_assist.devices import Exoskeleton


class TestGaitSimulator(unittest.TestCase):
    """Test the GaitSimulator class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.simulator = gsa.GaitSimulator()
    
    def test_create_default_model(self):
        """Test that a default model can be created."""
        self.assertIsNotNone(self.simulator.model)
        self.assertEqual(self.simulator.model.getName(), 'walk2d')
    
    def test_cost_function_creation(self):
        """Test that cost functions can be created."""
        cot = CostOfTransport(self.simulator.model)
        effort = MuscleEffort(self.simulator.model)
        hybrid = Hybrid(self.simulator.model, cost_functions={
            'cot': 0.5,
            'muscle_effort': 0.5
        })
        
        self.assertIsNotNone(cot)
        self.assertIsNotNone(effort)
        self.assertIsNotNone(hybrid)
    
    def test_exoskeleton_creation(self):
        """Test that an exoskeleton can be created."""
        exo = Exoskeleton(
            name="test_exo",
            model=self.simulator.model,
            joint_name="ankle",
            mass=1.0,
            max_torque=50.0
        )
        
        self.assertIsNotNone(exo)
        self.assertEqual(exo.name, "test_exo")
        self.assertEqual(exo.joint_name, "ankle")
        self.assertEqual(exo.mass, 1.0)
        self.assertEqual(exo.max_torque, 50.0)


if __name__ == '__main__':
    unittest.main() 