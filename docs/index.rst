GaitSim Assist Documentation
==========================

GaitSim Assist is a comprehensive Python library for biomechanics researchers and wearable device engineers. It provides a high-level API for running gait simulations, testing assistive device parameters, and optimizing cost function settings without deep diving into low-level OpenSim code.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   tutorials/index
   api/index
   examples
   contributing
   changelog

Features
--------

- **Unified Gait Simulation API**: High-level Python APIs to load gait datasets, run simulations (tracking or predictive), and retrieve joint kinematics & kinetics results.
- **Cost Function Modules**: Built-in library of cost functions (metabolic cost, fatigue, effort, etc.) that can be swapped or combined to see how assistive outcomes change.
- **Assistive Device Modeling**: Tools to model simple assistive device effects (like exoskeleton torques or prosthetic joint limits) and incorporate them into simulations.
- **Visualization & Analysis**: Automatic generation of gait plots – joint angle curves, ground reaction forces, muscle activations – for both experimental and synthetic gait data.

Quick Start
----------

.. code-block:: python

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 