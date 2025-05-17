# GaitSim Assist Library Summary

## Overview

GaitSim Assist is a Python library designed for biomechanics researchers and wearable device engineers. It provides a high-level API for running gait simulations, testing assistive device parameters, and optimizing cost function settings without deep diving into low-level OpenSim code.

## Library Structure

The library is organized into the following modules:

- **simulation**: Core simulation capabilities (GaitSimulator, TrackingSimulation, PredictiveSimulation)
- **cost_functions**: Various cost functions for optimization (MuscleEffort, JointTorque, Fatigue, HeadMotion, CostOfTransport, Hybrid)
- **devices**: Assistive device models (Exoskeleton, Prosthetic, Orthosis)
- **visualization**: Tools for visualizing results (GaitPlotter, plot_joint_angles, plot_ground_forces, plot_muscle_activations)
- **analysis**: Tools for analyzing results (calculate_gait_metrics, compare_simulations)

## Key Components

### SimulationResults

The `SimulationResults` class is a container for simulation results, including:
- Time points
- States (joint positions, velocities)
- Controls (actuator signals)
- Joint angles
- Ground reaction forces
- Muscle activations
- Metabolic cost
- Metrics (stride length, cadence, etc.)

### GaitSimulator

The `GaitSimulator` class is the main entry point for running simulations:
- `run_predictive_simulation()`: Run a predictive simulation with a specified cost function
- `run_tracking_simulation()`: Run a tracking simulation to follow reference data
- `export_results()`: Export simulation results to files

### Cost Functions

The library includes several cost functions:
- `MuscleEffort`: Minimizes muscle activations
- `JointTorque`: Minimizes joint torques
- `Fatigue`: Minimizes maximum muscle activation
- `HeadMotion`: Minimizes head acceleration
- `CostOfTransport`: Minimizes metabolic cost per distance traveled
- `Hybrid`: Combines multiple cost functions with weights

### Visualization

The `GaitPlotter` class provides methods for visualizing results:
- `plot_joint_angles()`: Plot joint angles over time or gait cycle
- `plot_ground_forces()`: Plot ground reaction forces
- `plot_muscle_activations()`: Plot muscle activations
- `compare_simulations()`: Compare results from multiple simulations

## Example Scripts

The library includes several example scripts:

1. **basic_simulation_moco.py**: Run a simple walking simulation using OpenSim Moco
2. **compare_cost_functions.py**: Compare different cost functions (muscle effort, joint torque, hybrid)
3. **analyze_opensim_results.py**: Analyze and visualize existing OpenSim simulation results
4. **batch_processing.py**: Run multiple simulations with different parameters
5. **assistive_device_optimization.py**: Optimize assistive device parameters
6. **custom_cost_function.py**: Create and use custom cost functions

## Getting Started

1. Install the library:
```bash
pip install -e .
```

2. Run an example:
```bash
cd examples
python basic_simulation_moco.py
```

3. Use the library in your own code:
```python
import gaitsim_assist as gsa

# Create a simulator with default 2D walking model
simulator = gsa.GaitSimulator()

# Run a predictive simulation with cost of transport cost function
results = simulator.run_predictive_simulation(
    cost_function='cot',
    time_range=(0.0, 1.0)
)

# Visualize the results
from gaitsim_assist.visualization import GaitPlotter
plotter = GaitPlotter()
plotter.plot_joint_angles(results)
plotter.plot_ground_forces(results)
```

## Notes

- The library requires OpenSim 4.3+ and Python 3.7+
- The example scripts use OpenSim Moco for optimization
- The library can be used with existing OpenSim models and results
- Custom cost functions can be created by extending the `CostFunction` base class

## Next Steps

1. Run the example scripts to understand how the library works
2. Explore the library's API in the source code
3. Try creating your own simulations with different cost functions
4. Experiment with adding assistive devices to the models 