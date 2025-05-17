# Models Directory

This directory contains OpenSim model files (.osim) used by GaitSim Assist.

## Default Models

- `walk2d.osim`: A simple 2D walking model with hip, knee, and ankle joints.

## Using Custom Models

You can use your own OpenSim models with GaitSim Assist by passing the path to the model file to the `GaitSimulator` constructor:

```python
import gaitsim_assist as gsa

# Create a simulator with a custom model
simulator = gsa.GaitSimulator(model_path="path/to/your/model.osim")
```

## Model Requirements

For a model to work with GaitSim Assist, it should:

1. Have at least one leg with hip, knee, and ankle joints
2. Have muscles or actuators for each joint
3. Have appropriate contact geometry for ground contact

## Contributing Models

If you have a model that you think would be useful for the community, please consider contributing it to the GaitSim Assist repository. See the main README for contribution guidelines. 