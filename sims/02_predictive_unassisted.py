"""
Predictive unassisted simulation script.

This script uses MocoStudy to predict unassisted gait with different
cost functions, using tracking solutions as initial guesses.
"""

import os
import json
from opensim import Model
import moco
from utilities import load_model, create_moco_problem

def main():
    # Configuration
    config = {
        "model_path": "../models/planar18.osim",
        "tracking_dir": "../results/tracking",
        "cost_functions": [
            "cot",
            "muscle_effort",
            "joint_torque",
            "fatigue",
            "head_motion",
            "hybrid"
        ],
        "output_dir": "../results/predictive_unassisted"
    }
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Load model
    model = load_model(config["model_path"])
    
    # Run predictive simulations for each cost function
    for cost_func in config["cost_functions"]:
        print(f"Running predictive simulation with {cost_func} cost function...")
        
        # Load tracking solution for initial guess
        traj_path = os.path.join(
            config["tracking_dir"],
            f"{cost_func}_tracking.traj"
        )
        guess = moco.MocoTrajectory(traj_path)
        
        # Create predictive problem
        problem = moco.MocoProblem()
        problem.setModel(model)
        problem.setTimeBounds(0.0, 1.0)  # One gait cycle
        
        # Set cost function
        if cost_func == "cot":
            problem.setCostFunction(CostOfTransport(model))
        elif cost_func == "muscle_effort":
            problem.setCostFunction(MuscleEffort(model))
        elif cost_func == "joint_torque":
            problem.setCostFunction(JointTorque(model))
        elif cost_func == "fatigue":
            problem.setCostFunction(Fatigue(model))
        elif cost_func == "head_motion":
            problem.setCostFunction(HeadMotion(model))
        elif cost_func == "hybrid":
            problem.setCostFunction(Hybrid(model))
        
        # Add constraints
        # Speed constraint
        speed_goal = moco.MocoAverageSpeedGoal()
        speed_goal.setTargetSpeed(1.25)  # m/s
        problem.addGoal(speed_goal)
        
        # Periodicity constraint
        periodicity = moco.MocoPeriodicityGoal()
        problem.addGoal(periodicity)
        
        # Ground clearance constraint
        clearance = moco.MocoMinimumHeightGoal()
        clearance.setHeight(0.02)  # 2 cm
        problem.addGoal(clearance)
        
        # Configure solver
        solver = moco.MocoCasADiSolver()
        solver.setProblem(problem)
        solver.set_num_mesh_points(100)
        solver.set_optim_convergence_tolerance(1e-3)
        solver.set_optim_constraint_tolerance(1e-3)
        
        # Set initial guess
        solver.setGuess(guess)
        
        # Solve
        solution = solver.solve()
        
        # Save results
        output_path = os.path.join(
            config["output_dir"],
            f"{cost_func}_predictive.sto"
        )
        solution.write(output_path)
        
        print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main() 