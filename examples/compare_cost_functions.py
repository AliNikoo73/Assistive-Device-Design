#!/usr/bin/env python3
"""
Example script to compare different cost functions for gait simulation.

This example demonstrates how to:
1. Run simulations with different cost functions
2. Compare and visualize the results
3. Analyze the differences between cost functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import opensim as osim
import opensim.moco as moco

import gaitsim_assist as gsa
from gaitsim_assist.visualization import GaitPlotter
from gaitsim_assist.cost_functions import MuscleEffort, JointTorque, Hybrid
from gaitsim_assist.analysis import compare_simulations, calculate_gait_metrics

# Create output directory
output_dir = Path("cost_function_comparison_results")
output_dir.mkdir(exist_ok=True)


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


def setup_moco_problem(model, cost_function):
    """Set up a Moco problem for walking with the specified cost function."""
    # Create a MocoStudy
    study = moco.MocoStudy()
    study.setName(f"walking_simulation_{cost_function.__class__.__name__}")
    
    # Get the problem
    problem = study.updProblem()
    
    # Set the model
    problem.setModelProcessor(osim.ModelProcessor(model))
    
    # Set the time range
    problem.setTimeBounds(0, 1.0)
    
    # Set the control bounds
    problem.setControlInfo("/hip_flexion_actuator", [-100, 100])
    problem.setControlInfo("/knee_flexion_actuator", [-100, 100])
    problem.setControlInfo("/ankle_flexion_actuator", [-100, 100])
    
    # Set the state bounds
    problem.setStateInfo("/hip_flexion/value", [-0.5, 1.0])
    problem.setStateInfo("/knee_flexion/value", [-1.5, 0.0])
    problem.setStateInfo("/ankle_flexion/value", [-0.5, 0.5])
    
    # Set the initial and final state bounds for pelvis translation
    problem.setStateInfo("/pelvis_tx/value", [0, 1.0], 0, 1.0)
    problem.setStateInfo("/pelvis_ty/value", [0.8, 1.0])
    
    # Add a periodic constraint for the joints (cyclic gait)
    periodic = moco.MocoPeriodicConstraint()
    periodic.setName("periodic_constraint")
    periodic.addStatePair("/hip_flexion/value")
    periodic.addStatePair("/knee_flexion/value")
    periodic.addStatePair("/ankle_flexion/value")
    periodic.addStatePair("/hip_flexion/speed")
    periodic.addStatePair("/knee_flexion/speed")
    periodic.addStatePair("/ankle_flexion/speed")
    problem.addPathConstraint(periodic)
    
    # Add the specified cost function
    if isinstance(cost_function, MuscleEffort):
        effort = moco.MocoControlGoal("muscle_effort")
        effort.setWeight(10)
        problem.addGoal(effort)
    elif isinstance(cost_function, JointTorque):
        torque = moco.MocoControlGoal("joint_torque")
        torque.setExponent(2)
        torque.setWeight(1)
        problem.addGoal(torque)
    elif isinstance(cost_function, Hybrid):
        # Add both muscle effort and joint torque with different weights
        effort = moco.MocoControlGoal("muscle_effort")
        effort.setWeight(5)
        problem.addGoal(effort)
        
        torque = moco.MocoControlGoal("joint_torque")
        torque.setExponent(2)
        torque.setWeight(0.5)
        problem.addGoal(torque)
    
    # Add a goal to track a reference speed
    speed_tracking = moco.MocoAverageSpeedGoal("speed_tracking")
    speed_tracking.set_desired_average_speed(1.0)  # 1 m/s
    speed_tracking.setWeight(100)
    problem.addGoal(speed_tracking)
    
    # Configure the solver
    solver = study.initCasADiSolver()
    solver.set_num_mesh_intervals(50)
    solver.set_optim_convergence_tolerance(1e-3)
    solver.set_optim_constraint_tolerance(1e-3)
    
    return study


def run_simulation_with_cost_function(cost_function):
    """Run a walking simulation with the specified cost function."""
    print(f"Running simulation with {cost_function.__class__.__name__} cost function...")
    
    model = create_simple_model()
    study = setup_moco_problem(model, cost_function)
    
    print("Solving the problem...")
    solution = study.solve()
    
    if not solution.success():
        print(f"Warning: Moco solution with {cost_function.__class__.__name__} was not successful.")
    
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
    
    # Add cost function specific metrics
    if isinstance(cost_function, MuscleEffort):
        # Calculate total muscle effort
        metrics['total_muscle_effort'] = np.sum([np.sum(muscle_activations[m]**2) for m in muscle_activations])
    elif isinstance(cost_function, JointTorque):
        # Calculate total joint torque
        joint_torques = {
            'hip': np.abs(controls.get('hip_flexion_actuator', np.zeros_like(time))),
            'knee': np.abs(controls.get('knee_flexion_actuator', np.zeros_like(time))),
            'ankle': np.abs(controls.get('ankle_flexion_actuator', np.zeros_like(time)))
        }
        metrics['total_joint_torque'] = np.sum([np.sum(joint_torques[j]**2) for j in joint_torques])
    
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
    
    return results


def main():
    # Create a simple model for cost function initialization
    model = create_simple_model()
    
    # Create cost functions
    muscle_effort_cost = MuscleEffort(model)
    joint_torque_cost = JointTorque(model)
    hybrid_cost = Hybrid(model, cost_functions={'muscle_effort': 0.7, 'joint_torque': 0.3})
    
    # Run simulations with different cost functions
    muscle_effort_results = run_simulation_with_cost_function(muscle_effort_cost)
    joint_torque_results = run_simulation_with_cost_function(joint_torque_cost)
    hybrid_results = run_simulation_with_cost_function(hybrid_cost)
    
    # Create a plotter
    plotter = GaitPlotter()
    
    # Compare joint angles between different cost functions
    print("Comparing joint angles...")
    plotter.compare_simulations(
        results_list=[muscle_effort_results, joint_torque_results, hybrid_results],
        labels=["Muscle Effort", "Joint Torque", "Hybrid"],
        plot_type="joint_angles",
        normalize_gait_cycle=True,
        save_path=output_dir / "comparison_joint_angles.png"
    )
    
    # Compare muscle activations between different cost functions
    print("Comparing muscle activations...")
    plotter.compare_simulations(
        results_list=[muscle_effort_results, joint_torque_results, hybrid_results],
        labels=["Muscle Effort", "Joint Torque", "Hybrid"],
        plot_type="muscle_activations",
        normalize_gait_cycle=True,
        save_path=output_dir / "comparison_muscle_activations.png"
    )
    
    # Compare ground reaction forces between different cost functions
    print("Comparing ground reaction forces...")
    plotter.compare_simulations(
        results_list=[muscle_effort_results, joint_torque_results, hybrid_results],
        labels=["Muscle Effort", "Joint Torque", "Hybrid"],
        plot_type="ground_forces",
        normalize_gait_cycle=True,
        save_path=output_dir / "comparison_ground_forces.png"
    )
    
    # Calculate and compare gait metrics
    print("Comparing gait metrics...")
    metrics_df = compare_simulations(
        results_list=[muscle_effort_results, joint_torque_results, hybrid_results],
        labels=["Muscle Effort", "Joint Torque", "Hybrid"],
        plot=True,
        save_path=output_dir / "metrics_comparison.png"
    )
    
    # Save metrics to CSV
    metrics_df.to_csv(output_dir / "metrics_comparison.csv")
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main() 