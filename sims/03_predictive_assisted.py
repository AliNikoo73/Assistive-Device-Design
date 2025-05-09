"""
Predictive assisted simulation script.

This script uses MocoStudy to predict assisted gait with different
cost functions and assistive devices, using unassisted solutions as
initial guesses.
"""

import os
import json
from opensim import Model
import moco
from utilities import load_model, create_moco_problem, add_device

def main():
    # Configuration
    config = {
        "model_path": "../models/planar18.osim",
        "unassisted_dir": "../results/predictive_unassisted",
        "cost_functions": [
            "cot",
            "muscle_effort",
            "joint_torque",
            "fatigue",
            "head_motion",
            "hybrid"
        ],
        "device_locations": ["hip", "knee", "ankle"],
        "output_dir": "../results/predictive_assisted"
    }
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Run predictive simulations for each cost function and device location
    for cost_func in config["cost_functions"]:
        for device_loc in config["device_locations"]:
            print(f"Running predictive simulation with {cost_func} cost function "
                  f"and {device_loc} device...")
            
            # Load unassisted solution for initial guess
            guess_path = os.path.join(
                config["unassisted_dir"],
                f"{cost_func}_predictive.sto"
            )
            guess = moco.MocoTrajectory(guess_path)
            
            # Load and modify model
            model = load_model(config["model_path"])
            add_device(model, device_loc)
            
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
                f"{cost_func}_{device_loc}_assisted.sto"
            )
            solution.write(output_path)
            
            print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main() 