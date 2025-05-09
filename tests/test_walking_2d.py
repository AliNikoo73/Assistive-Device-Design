import pytest
import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sims.walking_2d import Walking2DSimulation

def test_model_creation():
    """Test if the model can be created successfully"""
    sim = Walking2DSimulation()
    model = sim.create_model()
    
    # Check if model was created
    assert model is not None
    assert sim.model is not None
    
    # Check if basic components exist
    assert model.getBodySet().getSize() > 0
    assert model.getJointSet().getSize() > 0
    assert len(sim.contact_forces) > 0

def test_muscle_setup():
    """Test if muscles are properly added to the model"""
    sim = Walking2DSimulation()
    model = sim.create_model()
    sim.setup_muscles()
    
    # Check if muscles exist
    muscles = ['hip_flexor_r', 'hip_extensor_r', 
              'knee_extensor_r', 'knee_flexor_r',
              'ankle_plantar_r', 'ankle_dorsi_r']
              
    for muscle in muscles:
        assert model.getMuscles().get(muscle) is not None

def test_metabolics_setup():
    """Test if metabolics probe is properly added"""
    sim = Walking2DSimulation()
    model = sim.create_model()
    sim.setup_muscles()
    sim.setup_metabolics()
    
    # Check if probe exists
    assert model.getProbeSet().getSize() > 0

def test_study_creation():
    """Test if the optimization study is properly configured"""
    sim = Walking2DSimulation()
    model = sim.create_model()
    sim.setup_muscles()
    sim.setup_metabolics()
    study = sim.create_study()
    
    # Check if study was created
    assert study is not None
    assert sim.study is not None
    
    # Check if problem is configured
    problem = study.updProblem()
    assert problem is not None

@pytest.mark.skip(reason="This test runs a full simulation and may take several minutes")
def test_full_simulation():
    """Test a complete simulation run"""
    sim = Walking2DSimulation()
    model = sim.create_model()
    sim.setup_muscles()
    sim.setup_metabolics()
    study = sim.create_study()
    
    # Run simulation
    solution = sim.solve()
    assert solution is not None
    
    # Check if results directory is created
    results_dir = Path('results')
    assert results_dir.exists()
    
    # Analyze results
    metrics = sim.analyze_results(solution)
    assert metrics is not None
    
    # Check if visualization files are created
    sim.visualize(solution)
    assert (results_dir / 'simulation_results.png').exists()
    assert (results_dir / 'simulation_results.csv').exists()

def test_model_export():
    """Test if the model can be exported"""
    sim = Walking2DSimulation()
    model = sim.create_model()
    
    # Export model
    sim.export_model('test_model.osim')
    
    # Check if file exists
    assert Path('Models/test_model.osim').exists()
    
    # Clean up
    Path('Models/test_model.osim').unlink()

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 