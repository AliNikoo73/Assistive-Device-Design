# Data Directory

This directory contains data files used by GaitSim Assist, including:

- Reference gait data for validation
- Default parameter sets for simulations
- Example results for testing

## Reference Gait Data

Reference gait data is stored in CSV files with the following columns:

- `time`: Time in seconds
- `hip_angle`, `knee_angle`, `ankle_angle`: Joint angles in degrees
- `ground_force_x`, `ground_force_y`: Ground reaction forces in Newtons
- `hip_flexor_emg`, `hip_extensor_emg`, etc.: EMG data for various muscles

## Parameter Sets

Parameter sets are stored in JSON files with the following structure:

```json
{
  "simulation": {
    "time_range": [0.0, 1.0],
    "time_step": 0.01
  },
  "cost_function": {
    "name": "cot",
    "weight": 1.0
  },
  "device": {
    "type": "exoskeleton",
    "joint_name": "ankle",
    "mass": 1.0,
    "max_torque": 50.0
  }
}
```

## Using Custom Data

You can use your own data files with GaitSim Assist by passing the path to the data file to the appropriate function:

```python
import gaitsim_assist as gsa

# Create a simulator
simulator = gsa.GaitSimulator()

# Run a tracking simulation with custom reference data
results = simulator.run_tracking_simulation(
    reference_data="path/to/your/reference_data.csv",
    time_range=(0.0, 1.0)
) 