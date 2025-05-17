# GaitSim Assist Examples

This directory contains example scripts demonstrating how to use the GaitSim Assist library for gait simulation and analysis.

## Basic Examples

### Basic Simulation with Moco

**Script:** `basic_simulation_moco.py`

This example demonstrates how to:
1. Create a simple 2D walking model
2. Set up a Moco tracking problem
3. Solve the problem and visualize results

To run:
```bash
python basic_simulation_moco.py
```

The script will create a `simulation_results` directory with plots of joint angles, ground reaction forces, and muscle activations.

### Compare Cost Functions

**Script:** `compare_cost_functions.py`

This example demonstrates how to:
1. Run simulations with different cost functions (muscle effort, joint torque, hybrid)
2. Compare and visualize the results
3. Analyze the differences between cost functions

To run:
```bash
python compare_cost_functions.py
```

The script will create a `cost_function_comparison_results` directory with comparison plots and metrics.

### Analyze OpenSim Results

**Script:** `analyze_opensim_results.py`

This example demonstrates how to:
1. Load OpenSim simulation results
2. Convert them to GaitSim Assist format
3. Analyze and visualize the results

To run:
```bash
python analyze_opensim_results.py
```

The script expects OpenSim result files in an `opensim_results` directory:
- `states.sto`: State trajectories
- `forces.sto`: Ground reaction forces (optional)
- `controls.sto`: Control signals and muscle activations (optional)

The script will create an `opensim_analysis_results` directory with plots and metrics.

## Advanced Examples

### Batch Processing

**Script:** `batch_processing.py`

This example demonstrates how to:
1. Run multiple simulations with different parameters
2. Process and analyze the results in batch
3. Generate summary statistics and plots

To run:
```bash
python batch_processing.py
```

### Assistive Device Optimization

**Script:** `assistive_device_optimization.py`

This example demonstrates how to:
1. Add an assistive device to a model
2. Optimize device parameters
3. Compare assisted and unassisted gait

To run:
```bash
python assistive_device_optimization.py
```

### Custom Cost Function

**Script:** `custom_cost_function.py`

This example demonstrates how to:
1. Create a custom cost function
2. Integrate it with the GaitSim Assist framework
3. Use it in a simulation

To run:
```bash
python custom_cost_function.py
```

## Requirements

These examples require:
- Python 3.7+
- OpenSim 4.3+
- GaitSim Assist library

Make sure you have installed the GaitSim Assist library:
```bash
pip install -e ..
```

## Running Examples

To run an example, make sure GaitSim Assist is installed, then run:

```bash
python examples/basic_simulation.py
```

## Output

Most examples save their output to the `simulation_results` directory. This includes:

- CSV files with simulation data
- PNG files with visualizations
- JSON files with metrics and parameters 