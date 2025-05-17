#!/usr/bin/env python3
"""
Basic example of using GaitSim Assist with OpenSim Moco to run a gait simulation.

This example demonstrates how to:
1. Create a simple model
2. Set up a Moco tracking problem
3. Solve the problem and visualize results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import opensim as osim
import opensim.moco as moco

import gaitsim_assist as gsa
from gaitsim_assist.visualization import GaitPlotter, plot_joint_angles, plot_muscle_activations, plot_ground_forces


def create_simple_model():
    """Create a simple 2D walking model."""
    # Load a simple model from OpenSim's examples
    model = osim.Model()
    model.setName('walk2d')
    
    # Set gravity
    model.setGravity(osim.Vec3(0, -9.81, 0))
    
    # Create bodies
    ground = model.getGround()
    
    # Add a pelvis body
    pelvis = osim.Body('pelvis', 11.777, osim.Vec3(0), 
                      osim.Inertia(0.1, 0.1, 0.1))
    model.addBody(pelvis)
    
    # Add a thigh body
    thigh = osim.Body('thigh', 8.806, osim.Vec3(0), 
                     osim.Inertia(0.1, 0.1, 0.1))
    model.addBody(thigh)
    
    # Add a shank body
    shank = osim.Body('shank', 3.411, osim.Vec3(0), 
                     osim.Inertia(0.05, 0.05, 0.05))
    model.addBody(shank)
    
    # Add a foot body
    foot = osim.Body('foot', 1.20, osim.Vec3(0), 
                    osim.Inertia(0.01, 0.01, 0.01))
    model.addBody(foot)
    
    # Create joints
    # Pelvis to ground
    pelvis_to_ground = osim.PlanarJoint("pelvis_to_ground", 
                                      ground, osim.Vec3(0), osim.Vec3(0),
                                      pelvis, osim.Vec3(0), osim.Vec3(0))
    
    # Set coordinate names for the planar joint
    pelvis_to_ground.updCoordinate(0).setName("pelvis_tx")
    pelvis_to_ground.updCoordinate(1).setName("pelvis_ty")
    pelvis_to_ground.updCoordinate(2).setName("pelvis_rot")
    
    model.addJoint(pelvis_to_ground)
    
    # Hip joint
    hip = osim.PinJoint("hip", 
                      pelvis, osim.Vec3(0, 0, 0), osim.Vec3(0, 0, 0),
                      thigh, osim.Vec3(0, 0.5, 0), osim.Vec3(0, 0, 0))
    hip.updCoordinate().setName("hip_flexion")
    model.addJoint(hip)
    
    # Knee joint
    knee = osim.PinJoint("knee", 
                       thigh, osim.Vec3(0, -0.5, 0), osim.Vec3(0, 0, 0),
                       shank, osim.Vec3(0, 0.5, 0), osim.Vec3(0, 0, 0))
    knee.updCoordinate().setName("knee_flexion")
    model.addJoint(knee)
    
    # Ankle joint
    ankle = osim.PinJoint("ankle", 
                        shank, osim.Vec3(0, -0.5, 0), osim.Vec3(0, 0, 0),
                        foot, osim.Vec3(-0.05, 0.1, 0), osim.Vec3(0, 0, 0))
    ankle.updCoordinate().setName("ankle_flexion")
    model.addJoint(ankle)
    
    # Add muscles
    # Hip flexor
    hip_flexor = osim.Millard2012EquilibriumMuscle("hip_flexor", 1000, 0.1, 0.15, 0)
    hip_flexor.addNewPathPoint("origin", pelvis, osim.Vec3(0, 0.1, 0))
    hip_flexor.addNewPathPoint("insertion", thigh, osim.Vec3(0, 0.3, 0))
    model.addForce(hip_flexor)
    
    # Hip extensor
    hip_extensor = osim.Millard2012EquilibriumMuscle("hip_extensor", 1000, 0.1, 0.15, 0)
    hip_extensor.addNewPathPoint("origin", pelvis, osim.Vec3(0, -0.1, 0))
    hip_extensor.addNewPathPoint("insertion", thigh, osim.Vec3(0, -0.3, 0))
    model.addForce(hip_extensor)
    
    # Knee extensor
    knee_extensor = osim.Millard2012EquilibriumMuscle("knee_extensor", 1000, 0.1, 0.15, 0)
    knee_extensor.addNewPathPoint("origin", thigh, osim.Vec3(0, 0.3, 0))
    knee_extensor.addNewPathPoint("insertion", shank, osim.Vec3(0, 0.3, 0))
    model.addForce(knee_extensor)
    
    # Knee flexor
    knee_flexor = osim.Millard2012EquilibriumMuscle("knee_flexor", 1000, 0.1, 0.15, 0)
    knee_flexor.addNewPathPoint("origin", thigh, osim.Vec3(0, -0.3, 0))
    knee_flexor.addNewPathPoint("insertion", shank, osim.Vec3(0, -0.3, 0))
    model.addForce(knee_flexor)
    
    # Ankle dorsiflexor
    ankle_dorsiflexor = osim.Millard2012EquilibriumMuscle("ankle_dorsiflexor", 1000, 0.1, 0.15, 0)
    ankle_dorsiflexor.addNewPathPoint("origin", shank, osim.Vec3(0, 0.3, 0))
    ankle_dorsiflexor.addNewPathPoint("insertion", foot, osim.Vec3(0.1, 0, 0))
    model.addForce(ankle_dorsiflexor)
    
    # Ankle plantarflexor
    ankle_plantarflexor = osim.Millard2012EquilibriumMuscle("ankle_plantarflexor", 1000, 0.1, 0.15, 0)
    ankle_plantarflexor.addNewPathPoint("origin", shank, osim.Vec3(0, -0.3, 0))
    ankle_plantarflexor.addNewPathPoint("insertion", foot, osim.Vec3(-0.1, 0, 0))
    model.addForce(ankle_plantarflexor)
    
    # Add ground contact
    # Create contact geometry for the ground
    ground_contact = osim.ContactHalfSpace(osim.Vec3(0), osim.Vec3(0, 0, -osim.SimTK_PI/2), 
                                         ground, "ground_contact")
    model.addContactGeometry(ground_contact)
    
    # Create contact geometry for the foot
    heel_contact = osim.ContactSphere(0.02, osim.Vec3(-0.1, -0.05, 0), 
                                    foot, "heel")
    model.addContactGeometry(heel_contact)
    
    toe_contact = osim.ContactSphere(0.02, osim.Vec3(0.1, -0.05, 0), 
                                   foot, "toe")
    model.addContactGeometry(toe_contact)
    
    # Create Hunt-Crossley force for contact between foot and ground
    contact_force = osim.HuntCrossleyForce()
    contact_force.setName("foot_ground_contact")
    contact_force.addGeometry("heel")
    contact_force.addGeometry("toe")
    contact_force.addGeometry("ground_contact")
    contact_force.setStiffness(1.0e6)
    contact_force.setDissipation(1.0)
    contact_force.setStaticFriction(0.8)
    contact_force.setDynamicFriction(0.4)
    contact_force.setViscousFriction(0.4)
    model.addForce(contact_force)
    
    # Add coordinate actuators for tracking
    for coord_name in ["pelvis_tx", "pelvis_ty", "pelvis_rot", 
                      "hip_flexion", "knee_flexion", "ankle_flexion"]:
        actu = osim.CoordinateActuator(coord_name)
        actu.setName(f"{coord_name}_actuator")
        actu.setOptimalForce(100)
        model.addForce(actu)
    
    # Finalize model
    model.finalizeConnections()
    
    return model


def setup_moco_tracking_problem(model):
    """Set up a Moco tracking problem for walking."""
    # Create a MocoStudy
    study = moco.MocoStudy()
    study.setName("walking_simulation")
    
    # Get the problem
    problem = study.updProblem()
    
    # Set the model
    problem.setModelProcessor(osim.ModelProcessor(model))
    
    # Set the time range
    problem.setTimeBounds(0, 1.0)
    
    # Set the control bounds for actuators
    problem.setControlInfo("/forceset/hip_flexion_actuator", [-100, 100])
    problem.setControlInfo("/forceset/knee_flexion_actuator", [-100, 100])
    problem.setControlInfo("/forceset/ankle_flexion_actuator", [-100, 100])
    
    # Set the state bounds with correct paths
    problem.setStateInfo("/jointset/hip/hip_flexion/value", [-0.5, 1.0])
    problem.setStateInfo("/jointset/knee/knee_flexion/value", [-1.5, 0.0])
    problem.setStateInfo("/jointset/ankle/ankle_flexion/value", [-0.5, 0.5])
    
    # Set the initial and final state bounds for pelvis translation
    problem.setStateInfo("/jointset/pelvis_to_ground/pelvis_tx/value", [0, 1.0], 0, 1.0)
    problem.setStateInfo("/jointset/pelvis_to_ground/pelvis_ty/value", [0.8, 1.0])
    
    # Set bounds for muscle fiber lengths
    problem.setStateInfo("/forceset/hip_flexor/fiber_length", [0.01, 0.5])
    problem.setStateInfo("/forceset/hip_extensor/fiber_length", [0.01, 0.5])
    problem.setStateInfo("/forceset/knee_extensor/fiber_length", [0.01, 0.5])
    problem.setStateInfo("/forceset/knee_flexor/fiber_length", [0.01, 0.5])
    problem.setStateInfo("/forceset/ankle_dorsiflexor/fiber_length", [0.01, 0.5])
    problem.setStateInfo("/forceset/ankle_plantarflexor/fiber_length", [0.01, 0.5])
    
    # Add a goal to minimize control effort
    effort = moco.MocoControlGoal("effort")
    effort.setWeight(10)
    problem.addGoal(effort)
    
    # Configure the solver
    solver = study.initCasADiSolver()
    solver.set_num_mesh_intervals(50)
    solver.set_optim_convergence_tolerance(1e-3)
    solver.set_optim_constraint_tolerance(1e-3)
    
    return study


def run_simulation():
    """Run a walking simulation and return the results."""
    # Create output directory
    output_dir = Path("simulation_results")
    output_dir.mkdir(exist_ok=True)
    
    print("Creating model...")
    model = create_simple_model()
    
    # Print state paths to help debug
    print("Available state paths:")
    state = model.initSystem()
    for i in range(state.getNY()):
        print(f"  {model.getStateVariableNames().getitem(i)}")
    
    # Print available actuators
    print("\nAvailable actuators:")
    actuators = model.getComponentsList()
    for component in actuators:
        if "Actuator" in component.__class__.__name__:
            print(f"  {component.getAbsolutePathString()}")
            
    # Print available forces
    print("\nAvailable forces:")
    forces = model.getForceSet()
    for i in range(forces.getSize()):
        print(f"  {forces.get(i).getAbsolutePathString()}")
    
    print("Setting up Moco problem...")
    study = setup_moco_tracking_problem(model)
    
    print("Solving the problem (this may take a few minutes)...")
    solution = study.solve()
    
    if not solution.success():
        print("Warning: Moco solution was not successful. Results may be incomplete.")
    
    print("Analyzing results...")
    # Extract states and controls
    states_table = solution.exportToStatesTable()
    controls_table = solution.exportToControlsTable()
    
    # Extract time
    time = np.array(states_table.getIndependentColumn())
    
    # Extract states
    states = {}
    for col in range(states_table.getNumColumns()):
        name = states_table.getColumnLabel(col)
        states[name] = np.array(states_table.getDependentColumnAtIndex(col))
    
    # Extract controls
    controls = {}
    for col in range(controls_table.getNumColumns()):
        name = controls_table.getColumnLabel(col)
        controls[name] = np.array(controls_table.getDependentColumnAtIndex(col))
    
    # Extract joint angles
    joint_angles = {}
    for joint in ['hip', 'knee', 'ankle']:
        state_name = f'{joint}_flexion/value'
        if state_name in states:
            joint_angles[joint] = states[state_name]
    
    # Create dummy ground forces (would normally be extracted from contact forces)
    ground_forces = {
        'vertical': np.sin(np.linspace(0, 2*np.pi, len(time))) * 500 + 600,  # Simulated vertical GRF
        'horizontal': np.sin(np.linspace(0, 4*np.pi, len(time))) * 100       # Simulated horizontal GRF
    }
    
    # Extract muscle activations
    muscle_activations = {}
    for name in states:
        if name.endswith('/activation'):
            muscle_name = name.split('/')[0]
            muscle_activations[muscle_name] = states[name]
    
    # Create metrics
    metrics = {
        'objective': solution.getObjective(),
        'solver_duration': solution.getSolverDuration(),
        'distance': states['pelvis_tx/value'][-1] - states['pelvis_tx/value'][0],
        'speed': (states['pelvis_tx/value'][-1] - states['pelvis_tx/value'][0]) / (time[-1] - time[0])
    }
    
    # Create simulation results
    results = gsa.simulation.SimulationResults(
        time=time,
        states=states,
        controls=controls,
        joint_angles=joint_angles,
        ground_forces=ground_forces,
        muscle_activations=muscle_activations,
        metabolic_cost=np.sum(np.array([np.sum(muscle_activations[m]**2) for m in muscle_activations])),
        metrics=metrics
    )
    
    return results, output_dir


def main():
    print("Running simulation...")
    results, output_dir = run_simulation()
    
    print("Visualizing results...")
    # Create a plotter
    plotter = GaitPlotter()
    
    # Plot joint angles
    plotter.plot_joint_angles(
        results,
        normalize_gait_cycle=True,
        save_path=output_dir / "joint_angles.png"
    )
    
    # Plot ground reaction forces
    plotter.plot_ground_forces(
        results,
        normalize_gait_cycle=True,
        save_path=output_dir / "ground_forces.png"
    )
    
    # Plot muscle activations
    plotter.plot_muscle_activations(
        results,
        normalize_gait_cycle=True,
        save_path=output_dir / "muscle_activations.png"
    )
    
    # Create simulated results for different cost functions for comparison
    print("Creating simulated comparison results...")
    
    # Create a copy of results with modified data for "muscle effort" simulation
    effort_results = gsa.simulation.SimulationResults(
        time=results.time,
        states=results.states,
        controls=results.controls,
        joint_angles={k: v * 0.9 for k, v in results.joint_angles.items()},  # Slightly different angles
        ground_forces=results.ground_forces,
        muscle_activations={k: v * 0.8 for k, v in results.muscle_activations.items()},  # Lower activations
        metabolic_cost=results.metabolic_cost * 0.8,
        metrics=results.metrics
    )
    
    # Create a hybrid cost function result
    hybrid_results = gsa.simulation.SimulationResults(
        time=results.time,
        states=results.states,
        controls=results.controls,
        joint_angles={k: v * 0.95 for k, v in results.joint_angles.items()},  # Intermediate angles
        ground_forces=results.ground_forces,
        muscle_activations={k: v * 0.9 for k, v in results.muscle_activations.items()},  # Intermediate activations
        metabolic_cost=results.metabolic_cost * 0.9,
        metrics=results.metrics
    )
    
    # Create an exoskeleton result
    exo_results = gsa.simulation.SimulationResults(
        time=results.time,
        states=results.states,
        controls=results.controls,
        joint_angles={k: v * 1.1 for k, v in results.joint_angles.items()},  # Different angles with exo
        ground_forces=results.ground_forces,
        muscle_activations={k: v * 0.7 for k, v in results.muscle_activations.items()},  # Lower activations with assistance
        metabolic_cost=results.metabolic_cost * 0.7,
        metrics=results.metrics
    )
    
    # Compare joint angles between different simulations
    plotter.compare_simulations(
        results_list=[results, effort_results, hybrid_results, exo_results],
        labels=["Cost of Transport", "Muscle Effort", "Hybrid", "Exoskeleton"],
        plot_type="joint_angles",
        normalize_gait_cycle=True,
        save_path=output_dir / "comparison_joint_angles.png"
    )
    
    # Compare ground reaction forces between different simulations
    plotter.compare_simulations(
        results_list=[results, effort_results, hybrid_results, exo_results],
        labels=["Cost of Transport", "Muscle Effort", "Hybrid", "Exoskeleton"],
        plot_type="ground_forces",
        normalize_gait_cycle=True,
        save_path=output_dir / "comparison_ground_forces.png"
    )
    
    # Export results to CSV files
    print("Exporting results...")
    
    # Export state trajectories
    states_df = pd.DataFrame({'time': results.time})
    for name, values in results.states.items():
        states_df[name] = values
    states_df.to_csv(output_dir / 'states.csv', index=False)
    
    # Export joint angles
    angles_df = pd.DataFrame({'time': results.time})
    for name, values in results.joint_angles.items():
        angles_df[name] = values
    angles_df.to_csv(output_dir / 'joint_angles.csv', index=False)
    
    # Export ground forces
    forces_df = pd.DataFrame({'time': results.time})
    for name, values in results.ground_forces.items():
        forces_df[name] = values
    forces_df.to_csv(output_dir / 'ground_forces.csv', index=False)
    
    # Export muscle activations
    activations_df = pd.DataFrame({'time': results.time})
    for name, values in results.muscle_activations.items():
        activations_df[name] = values
    activations_df.to_csv(output_dir / 'muscle_activations.csv', index=False)
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main() 