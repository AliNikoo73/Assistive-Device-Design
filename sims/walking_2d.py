import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import json
from scipy import signal, interpolate
import seaborn as sns
from dataclasses import dataclass

@dataclass
class ExperimentalData:
    """Class to hold experimental gait data for validation"""
    time: np.ndarray
    joint_angles: Dict[str, np.ndarray]
    ground_forces: Dict[str, np.ndarray]
    emg: Dict[str, np.ndarray]
    
    @classmethod
    def from_file(cls, filepath: Path) -> 'ExperimentalData':
        """Load experimental data from a CSV file"""
        data = pd.read_csv(filepath)
        return cls(
            time=data['time'].values,
            joint_angles={
                'hip': data['hip_angle'].values,
                'knee': data['knee_angle'].values,
                'ankle': data['ankle_angle'].values
            },
            ground_forces={
                'vertical': data['ground_force_y'].values,
                'horizontal': data['ground_force_x'].values
            },
            emg={
                'hip_flexor': data['hip_flexor_emg'].values,
                'hip_extensor': data['hip_extensor_emg'].values,
                'knee_extensor': data['knee_extensor_emg'].values,
                'knee_flexor': data['knee_flexor_emg'].values,
                'ankle_plantar': data['ankle_plantar_emg'].values,
                'ankle_dorsi': data['ankle_dorsi_emg'].values
            }
        )

class Walking2DSimulation:
    def __init__(self):
        """Initialize the 2D walking simulation environment"""
        self.model = None
        self.study = None
        self.contact_forces = []
        self.experimental_data = None
        
    def load_experimental_data(self, filepath: Path):
        """Load experimental data for validation"""
        self.experimental_data = ExperimentalData.from_file(filepath)
        
    def create_model(self):
        """Create a 2D walking model with metabolics"""
        # Create a new OpenSim model
        model = osim.Model()
        model.setName('walk2d')
        
        # Set gravity
        model.setGravity(osim.Vec3(0, -9.81, 0))
        
        # Create ground reference frame
        ground = model.getGround()
        
        # Add a free joint to connect the pelvis to ground
        pelvis = osim.Body('pelvis', 11.777, osim.Vec3(0), 
                          osim.Inertia(0.0765, 0.0765, 0.0765))
        model.addBody(pelvis)
        
        free_joint = osim.FreeJoint('ground_pelvis', ground, pelvis)
        model.addJoint(free_joint)
        
        # Add thigh, shank, and foot segments
        thigh = osim.Body('thigh', 8.806, osim.Vec3(0), 
                         osim.Inertia(0.1, 0.1, 0.1))
        model.addBody(thigh)
        
        shank = osim.Body('shank', 3.411, osim.Vec3(0),
                         osim.Inertia(0.05, 0.05, 0.05))
        model.addBody(shank)
        
        foot = osim.Body('foot', 1.20, osim.Vec3(0),
                        osim.Inertia(0.01, 0.01, 0.01))
        model.addBody(foot)
        
        # Add hip, knee, and ankle joints
        hip = osim.PinJoint('hip', pelvis, osim.Vec3(0), osim.Vec3(0),
                           thigh, osim.Vec3(0, 0.5, 0), osim.Vec3(0))
        model.addJoint(hip)
        
        knee = osim.PinJoint('knee', thigh, osim.Vec3(0, -0.5, 0), osim.Vec3(0),
                            shank, osim.Vec3(0, 0.5, 0), osim.Vec3(0))
        model.addJoint(knee)
        
        ankle = osim.PinJoint('ankle', shank, osim.Vec3(0, -0.5, 0), osim.Vec3(0),
                             foot, osim.Vec3(-0.05, 0.1, 0), osim.Vec3(0))
        model.addJoint(ankle)
        
        # Add contact geometry to the ground
        ground_force = osim.HuntCrossleyForce()
        ground_force.setName('ground_force')
        ground_force.setStiffness(1e6)  # N/m
        ground_force.setDissipation(1.0)  # s/m
        ground_force.setStaticFriction(0.8)
        ground_force.setDynamicFriction(0.4)
        ground_force.setViscousFriction(0.4)
        
        # Create contact half-space for ground
        contact_ground = osim.ContactHalfSpace(osim.Vec3(0), osim.Vec3(0, 0, -np.pi/2), ground, "ground_contact")
        ground_force.addGeometry("ground_contact")
        
        # Add contact points to the foot
        heel_point = osim.ContactSphere(0.02, osim.Vec3(-0.05, -0.02, 0), foot, "heel")
        toe_point = osim.ContactSphere(0.02, osim.Vec3(0.05, -0.02, 0), foot, "toe")
        ground_force.addGeometry("heel")
        ground_force.addGeometry("toe")
        
        model.addForce(ground_force)
        self.contact_forces.append(ground_force)
        
        # Enhanced ground contact model
        self._add_enhanced_ground_contact(model)
        
        self.model = model
        return model
    
    def _add_enhanced_ground_contact(self, model: osim.Model):
        """Add enhanced ground contact model with multiple contact points and force profiles"""
        ground = model.getGround()
        foot = model.getBodySet().get("foot")
        
        # Create ground plane
        ground_force = osim.SmoothSphereHalfSpaceForce("ground_contact")
        ground_force.set_stiffness(1e6)  # N/m
        ground_force.set_dissipation(1.0)  # s/m
        ground_force.set_static_friction(0.8)
        ground_force.set_dynamic_friction(0.4)
        ground_force.set_transition_velocity(0.2)  # m/s
        
        # Add multiple contact points on foot
        contact_points = [
            ("heel", -0.05, -0.02),  # heel
            ("midfoot", 0.0, -0.02),  # midfoot
            ("toe", 0.05, -0.02)     # toe
        ]
        
        for name, x, y in contact_points:
            contact_sphere = osim.ContactSphere(
                0.02,  # radius
                osim.Vec3(x, y, 0),  # position
                foot,  # body
                name   # name
            )
            ground_force.addGeometry(name)
            
        # Add contact mesh for ground
        ground_mesh = osim.ContactHalfSpace(
            osim.Vec3(0),  # position
            osim.Vec3(0, 0, -np.pi/2),  # orientation
            ground,  # body
            "ground_mesh"  # name
        )
        ground_force.addGeometry("ground_mesh")
        
        # Add force to model
        model.addForce(ground_force)
        self.contact_forces.append(ground_force)
        
    def setup_muscles(self):
        """Add muscles to the model"""
        if self.model is None:
            raise ValueError("Model must be created first")
            
        # Define muscle parameters
        max_isometric_force = 4000  # N
        optimal_fiber_length = 0.10  # m
        tendon_slack_length = 0.15   # m
        pennation_angle = 0.0
        
        # Create hip flexor
        hip_flexor = osim.Thelen2003Muscle('hip_flexor_r', max_isometric_force,
                                         optimal_fiber_length, tendon_slack_length,
                                         pennation_angle)
        hip_flexor.addNewPathPoint('origin', self.model.getBody('pelvis'),
                                 osim.Vec3(0.0, 0.1, 0))
        hip_flexor.addNewPathPoint('insertion', self.model.getBody('thigh'),
                                 osim.Vec3(0.0, 0.4, 0))
        self.model.addForce(hip_flexor)
        
        # Create hip extensor
        hip_extensor = osim.Thelen2003Muscle('hip_extensor_r', max_isometric_force,
                                           optimal_fiber_length, tendon_slack_length,
                                           pennation_angle)
        hip_extensor.addNewPathPoint('origin', self.model.getBody('pelvis'),
                                   osim.Vec3(0.0, -0.1, 0))
        hip_extensor.addNewPathPoint('insertion', self.model.getBody('thigh'),
                                   osim.Vec3(0.0, -0.4, 0))
        self.model.addForce(hip_extensor)
        
        # Create knee extensor (quadriceps)
        knee_extensor = osim.Thelen2003Muscle('knee_extensor_r', max_isometric_force,
                                            optimal_fiber_length, tendon_slack_length,
                                            pennation_angle)
        knee_extensor.addNewPathPoint('origin', self.model.getBody('thigh'),
                                    osim.Vec3(0.0, 0.1, 0))
        knee_extensor.addNewPathPoint('insertion', self.model.getBody('shank'),
                                    osim.Vec3(0.0, 0.4, 0))
        self.model.addForce(knee_extensor)
        
        # Create knee flexor (hamstrings)
        knee_flexor = osim.Thelen2003Muscle('knee_flexor_r', max_isometric_force,
                                          optimal_fiber_length, tendon_slack_length,
                                          pennation_angle)
        knee_flexor.addNewPathPoint('origin', self.model.getBody('thigh'),
                                  osim.Vec3(0.0, -0.1, 0))
        knee_flexor.addNewPathPoint('insertion', self.model.getBody('shank'),
                                  osim.Vec3(0.0, -0.4, 0))
        self.model.addForce(knee_flexor)
        
        # Create ankle plantarflexor (gastrocnemius)
        ankle_plantar = osim.Thelen2003Muscle('ankle_plantar_r', max_isometric_force,
                                            optimal_fiber_length, tendon_slack_length,
                                            pennation_angle)
        ankle_plantar.addNewPathPoint('origin', self.model.getBody('shank'),
                                    osim.Vec3(-0.02, -0.15, 0))
        ankle_plantar.addNewPathPoint('insertion', self.model.getBody('foot'),
                                    osim.Vec3(-0.08, -0.01, 0))
        self.model.addForce(ankle_plantar)
        
        # Create ankle dorsiflexor (tibialis anterior)
        ankle_dorsi = osim.Thelen2003Muscle('ankle_dorsi_r', max_isometric_force,
                                          optimal_fiber_length, tendon_slack_length,
                                          pennation_angle)
        ankle_dorsi.addNewPathPoint('origin', self.model.getBody('shank'),
                                  osim.Vec3(0.02, -0.15, 0))
        ankle_dorsi.addNewPathPoint('insertion', self.model.getBody('foot'),
                                  osim.Vec3(0.06, -0.01, 0))
        self.model.addForce(ankle_dorsi)
        
    def setup_metabolics(self):
        """Add metabolics probe to the model"""
        if self.model is None:
            raise ValueError("Model must be created first")
            
        # Create Umberger2010MuscleMetabolicsProbe
        probe = osim.Umberger2010MuscleMetabolicsProbe()
        probe.setOperation("value")
        probe.set_report_total_metabolics_only(True)
        
        # Add muscles to be probed
        muscle_names = ['hip_flexor_r', 'hip_extensor_r', 
                       'knee_extensor_r', 'knee_flexor_r',
                       'ankle_plantar_r', 'ankle_dorsi_r']
        
        for muscle_name in muscle_names:
            probe.addMuscle(muscle_name)
        
        # Add to model
        self.model.addProbe(probe)
        
    def create_study(self):
        """Create a Moco study for optimization"""
        study = osim.MocoStudy()
        problem = study.updProblem()
        
        # Set the model
        problem.setModelCopy(self.model)
        
        # Set time bounds (typical gait cycle is about 1.2 seconds)
        problem.setTimeBounds(0, [0.8, 1.2])
        
        # Set optimization goals
        
        # 1. Minimize effort (muscle activation squared)
        effort = osim.MocoControlGoal("effort")
        effort.setWeight(10)
        problem.addGoal(effort)
        
        # 2. Minimize metabolic cost
        metabolics = osim.MocoOutputGoal("metabolics", 0.1)
        metabolics.setOutputPath("/metabolics")
        problem.addGoal(metabolics)
        
        # 3. Track normal walking speed
        speed = osim.MocoAverageSpeedGoal("speed")
        speed.set_desired_average_speed(1.3)  # typical walking speed in m/s
        speed.setWeight(2)
        problem.addGoal(speed)
        
        # Add periodic constraints for continuous walking
        periodic = osim.MocoPeriodicityGoal("periodicity")
        
        # States to make periodic (ensure end matches beginning)
        periodic_states = [
            "/jointset/hip/value",
            "/jointset/hip/speed",
            "/jointset/knee/value",
            "/jointset/knee/speed",
            "/jointset/ankle/value",
            "/jointset/ankle/speed"
        ]
        
        for state in periodic_states:
            periodic.addStatePair(osim.MocoPeriodicityGoalPair(state))
            
        # Add stride length constraint
        periodic.addStatePair(
            osim.MocoPeriodicityGoalPair("/jointset/ground_pelvis/pelvis_tx/value"),
            True)  # allow offset for forward progression
            
        problem.addGoal(periodic)
        
        # Set bounds on states
        problem.setStateInfo("/jointset/hip/value", [-0.5, 0.5])   # hip flexion angle
        problem.setStateInfo("/jointset/knee/value", [0, 1.5])     # knee flexion angle
        problem.setStateInfo("/jointset/ankle/value", [-0.5, 0.5]) # ankle angle
        
        # Set bounds on controls (muscle activations)
        for muscle in ['hip_flexor_r', 'hip_extensor_r', 
                      'knee_extensor_r', 'knee_flexor_r',
                      'ankle_plantar_r', 'ankle_dorsi_r']:
            problem.setControlInfo(f'/forceset/{muscle}/activation', [0, 1])
        
        # Configure the solver
        solver = study.initCasADiSolver()
        solver.set_num_mesh_intervals(50)  # number of time points
        solver.set_optim_solver("ipopt")   # interior point optimizer
        solver.set_optim_convergence_tolerance(1e-4)
        solver.set_optim_constraint_tolerance(1e-4)
        
        self.study = study
        return study
    
    def solve(self):
        """Solve the optimization problem"""
        if self.study is None:
            raise ValueError("Study must be created first")
            
        solver = self.study.initCasADiSolver()
        solution = solver.solve()
        return solution
    
    def visualize(self, solution):
        """Visualize the simulation results
        
        Args:
            solution: The MocoSolution object containing simulation results
        """
        if solution is None:
            raise ValueError("No solution provided for visualization")
            
        # Get time points
        time = solution.getTimeMat()
        
        # Create subplots for different metrics
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(3, 2)
        
        # 1. Joint Angles
        ax_angles = fig.add_subplot(gs[0, 0])
        hip_angle = solution.getStateMat('/jointset/hip/value')
        knee_angle = solution.getStateMat('/jointset/knee/value')
        ankle_angle = solution.getStateMat('/jointset/ankle/value')
        
        ax_angles.plot(time, np.rad2deg(hip_angle), label='Hip')
        ax_angles.plot(time, np.rad2deg(knee_angle), label='Knee')
        ax_angles.plot(time, np.rad2deg(ankle_angle), label='Ankle')
        ax_angles.set_xlabel('Time (s)')
        ax_angles.set_ylabel('Angle (degrees)')
        ax_angles.set_title('Joint Angles')
        ax_angles.legend()
        ax_angles.grid(True)
        
        # 2. Muscle Activations
        ax_activations = fig.add_subplot(gs[0, 1])
        muscles = ['hip_flexor_r', 'hip_extensor_r', 
                  'knee_extensor_r', 'knee_flexor_r',
                  'ankle_plantar_r', 'ankle_dorsi_r']
        
        for muscle in muscles:
            activation = solution.getControlMat(f'/forceset/{muscle}/activation')
            ax_activations.plot(time, activation, label=muscle)
        
        ax_activations.set_xlabel('Time (s)')
        ax_activations.set_ylabel('Activation')
        ax_activations.set_title('Muscle Activations')
        ax_activations.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_activations.grid(True)
        
        # 3. Ground Reaction Forces
        ax_grf = fig.add_subplot(gs[1, :])
        vertical_force = solution.getStateMat('/forceset/contactHeel_r/value_y')
        horizontal_force = solution.getStateMat('/forceset/contactHeel_r/value_x')
        
        ax_grf.plot(time, vertical_force, label='Vertical')
        ax_grf.plot(time, horizontal_force, label='Horizontal')
        ax_grf.set_xlabel('Time (s)')
        ax_grf.set_ylabel('Force (N)')
        ax_grf.set_title('Ground Reaction Forces')
        ax_grf.legend()
        ax_grf.grid(True)
        
        # 4. Metabolic Power
        ax_metabolics = fig.add_subplot(gs[2, :])
        metabolic_power = solution.getOutputMat('/metabolics')
        
        ax_metabolics.plot(time, metabolic_power, 'k-', label='Total')
        ax_metabolics.set_xlabel('Time (s)')
        ax_metabolics.set_ylabel('Metabolic Power (W)')
        ax_metabolics.set_title('Metabolic Power')
        ax_metabolics.legend()
        ax_metabolics.grid(True)
        
        # Adjust layout and display
        plt.tight_layout()
        
        # Save the figure
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / 'simulation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save numerical data
        results_data = {
            'time': time,
            'hip_angle': np.rad2deg(hip_angle),
            'knee_angle': np.rad2deg(knee_angle),
            'ankle_angle': np.rad2deg(ankle_angle),
            'vertical_force': vertical_force,
            'horizontal_force': horizontal_force,
            'metabolic_power': metabolic_power
        }
        
        # Add muscle activations to results
        for muscle in muscles:
            results_data[f'{muscle}_activation'] = solution.getControlMat(f'/forceset/{muscle}/activation')
            
        # Save to CSV
        df = pd.DataFrame(results_data)
        df.to_csv(results_dir / 'simulation_results.csv', index=False)
        
        print(f"Results saved to {results_dir}")

    def export_model(self, filename: str = 'walk2d_model.osim'):
        """Export the OpenSim model to a file
        
        Args:
            filename: Name of the file to save the model to
        """
        if self.model is None:
            raise ValueError("No model to export")
            
        models_dir = Path('Models')
        models_dir.mkdir(exist_ok=True)
        self.model.printToXML(str(models_dir / filename))
        print(f"Model exported to {models_dir / filename}")

    def analyze_results(self, solution) -> Dict:
        """Analyze the simulation results and generate metrics
        
        Args:
            solution: The MocoSolution object containing simulation results
            
        Returns:
            Dictionary containing analysis metrics
        """
        if solution is None:
            raise ValueError("No solution provided for analysis")
            
        # Get time points
        time = solution.getTimeMat()
        dt = time[1] - time[0]
        
        # Calculate key metrics
        metrics = {}
        
        # 1. Gait Cycle Analysis
        metrics['gait_cycle_duration'] = float(time[-1] - time[0])
        
        # 2. Joint Range of Motion (ROM)
        joint_angles = {
            'hip': np.rad2deg(solution.getStateMat('/jointset/hip/value')),
            'knee': np.rad2deg(solution.getStateMat('/jointset/knee/value')),
            'ankle': np.rad2deg(solution.getStateMat('/jointset/ankle/value'))
        }
        
        metrics['joint_rom'] = {
            joint: {
                'min': float(np.min(angles)),
                'max': float(np.max(angles)),
                'range': float(np.max(angles) - np.min(angles))
            }
            for joint, angles in joint_angles.items()
        }
        
        # 3. Peak Ground Reaction Forces
        vertical_force = solution.getStateMat('/forceset/contactHeel_r/value_y')
        horizontal_force = solution.getStateMat('/forceset/contactHeel_r/value_x')
        
        metrics['ground_reaction_forces'] = {
            'vertical_peak': float(np.max(vertical_force)),
            'horizontal_peak': float(np.max(np.abs(horizontal_force)))
        }
        
        # 4. Muscle Analysis
        muscles = ['hip_flexor_r', 'hip_extensor_r', 
                  'knee_extensor_r', 'knee_flexor_r',
                  'ankle_plantar_r', 'ankle_dorsi_r']
                  
        metrics['muscle_metrics'] = {}
        for muscle in muscles:
            activation = solution.getControlMat(f'/forceset/{muscle}/activation')
            metrics['muscle_metrics'][muscle] = {
                'peak_activation': float(np.max(activation)),
                'mean_activation': float(np.mean(activation)),
                'activation_time': float(np.sum(activation > 0.1) * dt)  # time spent activated
            }
            
        # 5. Metabolic Cost Analysis
        metabolic_power = solution.getOutputMat('/metabolics')
        metrics['metabolics'] = {
            'total_cost': float(np.trapz(metabolic_power, time)),  # total metabolic cost
            'mean_power': float(np.mean(metabolic_power)),
            'peak_power': float(np.max(metabolic_power))
        }
        
        # 6. Walking Speed Analysis
        pelvis_dx = solution.getStateMat('/jointset/ground_pelvis/pelvis_tx/speed')
        metrics['walking_speed'] = {
            'mean': float(np.mean(pelvis_dx)),
            'peak': float(np.max(pelvis_dx)),
            'min': float(np.min(pelvis_dx))
        }
        
        # Save analysis results
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / 'analysis_results.json', 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # Generate summary report
        self._generate_report(metrics, results_dir / 'analysis_report.txt')
        
        return metrics
        
    def _generate_report(self, metrics: Dict, output_file: Path):
        """Generate a human-readable report from the analysis metrics
        
        Args:
            metrics: Dictionary containing analysis metrics
            output_file: Path to save the report
        """
        report = []
        report.append("Walking Simulation Analysis Report")
        report.append("=" * 40 + "\n")
        
        # Gait Cycle
        report.append("1. Gait Cycle")
        report.append("-" * 20)
        report.append(f"Duration: {metrics['gait_cycle_duration']:.3f} seconds\n")
        
        # Joint ROM
        report.append("2. Joint Range of Motion (degrees)")
        report.append("-" * 20)
        for joint, rom in metrics['joint_rom'].items():
            report.append(f"{joint.title()}:")
            report.append(f"  Min: {rom['min']:.1f}")
            report.append(f"  Max: {rom['max']:.1f}")
            report.append(f"  Range: {rom['range']:.1f}\n")
        
        # Ground Reaction Forces
        report.append("3. Ground Reaction Forces (N)")
        report.append("-" * 20)
        grf = metrics['ground_reaction_forces']
        report.append(f"Peak Vertical: {grf['vertical_peak']:.1f}")
        report.append(f"Peak Horizontal: {grf['horizontal_peak']:.1f}\n")
        
        # Muscle Analysis
        report.append("4. Muscle Analysis")
        report.append("-" * 20)
        for muscle, data in metrics['muscle_metrics'].items():
            report.append(f"{muscle}:")
            report.append(f"  Peak Activation: {data['peak_activation']:.2f}")
            report.append(f"  Mean Activation: {data['mean_activation']:.2f}")
            report.append(f"  Activation Time: {data['activation_time']:.3f} s\n")
        
        # Metabolics
        report.append("5. Metabolic Cost")
        report.append("-" * 20)
        report.append(f"Total Cost: {metrics['metabolics']['total_cost']:.1f} J")
        report.append(f"Mean Power: {metrics['metabolics']['mean_power']:.1f} W")
        report.append(f"Peak Power: {metrics['metabolics']['peak_power']:.1f} W\n")
        
        # Walking Speed
        report.append("6. Walking Speed (m/s)")
        report.append("-" * 20)
        report.append(f"Mean: {metrics['walking_speed']['mean']:.2f}")
        report.append(f"Peak: {metrics['walking_speed']['peak']:.2f}")
        report.append(f"Min: {metrics['walking_speed']['min']:.2f}\n")
        
        # Save report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))

    def validate_results(self, solution) -> Dict:
        """Validate simulation results against experimental data
        
        Args:
            solution: The MocoSolution object containing simulation results
            
        Returns:
            Dictionary containing validation metrics
        """
        if self.experimental_data is None:
            raise ValueError("No experimental data loaded for validation")
            
        validation_metrics = {}
        
        # Resample simulation data to match experimental time points
        sim_time = solution.getTimeMat()
        exp_time = self.experimental_data.time
        
        # 1. Joint Angle Validation
        joint_rmse = {}
        joint_correlation = {}
        
        for joint in ['hip', 'knee', 'ankle']:
            sim_angle = np.rad2deg(solution.getStateMat(f'/jointset/{joint}/value'))
            exp_angle = self.experimental_data.joint_angles[joint]
            
            # Interpolate simulation data to experimental time points
            f = interpolate.interp1d(sim_time, sim_angle, bounds_error=False, fill_value="extrapolate")
            sim_angle_interp = f(exp_time)
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((sim_angle_interp - exp_angle)**2))
            correlation = np.corrcoef(sim_angle_interp, exp_angle)[0, 1]
            
            joint_rmse[joint] = float(rmse)
            joint_correlation[joint] = float(correlation)
            
        validation_metrics['joint_angles'] = {
            'rmse': joint_rmse,
            'correlation': joint_correlation
        }
        
        # 2. Ground Reaction Force Validation
        grf_rmse = {}
        grf_correlation = {}
        
        for direction, force_path in [('vertical', 'value_y'), ('horizontal', 'value_x')]:
            sim_force = solution.getStateMat(f'/forceset/contactHeel_r/{force_path}')
            exp_force = self.experimental_data.ground_forces[direction]
            
            # Interpolate
            f = interpolate.interp1d(sim_time, sim_force, bounds_error=False, fill_value="extrapolate")
            sim_force_interp = f(exp_time)
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((sim_force_interp - exp_force)**2))
            correlation = np.corrcoef(sim_force_interp, exp_force)[0, 1]
            
            grf_rmse[direction] = float(rmse)
            grf_correlation[direction] = float(correlation)
            
        validation_metrics['ground_forces'] = {
            'rmse': grf_rmse,
            'correlation': grf_correlation
        }
        
        # 3. Muscle Activation vs EMG Validation
        muscle_correlation = {}
        for muscle, emg_key in [
            ('hip_flexor_r', 'hip_flexor'),
            ('hip_extensor_r', 'hip_extensor'),
            ('knee_extensor_r', 'knee_extensor'),
            ('knee_flexor_r', 'knee_flexor'),
            ('ankle_plantar_r', 'ankle_plantar'),
            ('ankle_dorsi_r', 'ankle_dorsi')
        ]:
            sim_activation = solution.getControlMat(f'/forceset/{muscle}/activation')
            exp_emg = self.experimental_data.emg[emg_key]
            
            # Interpolate
            f = interpolate.interp1d(sim_time, sim_activation, bounds_error=False, fill_value="extrapolate")
            sim_activation_interp = f(exp_time)
            
            # Calculate correlation
            correlation = np.corrcoef(sim_activation_interp, exp_emg)[0, 1]
            muscle_correlation[muscle] = float(correlation)
            
        validation_metrics['muscle_activation'] = {
            'correlation': muscle_correlation
        }
        
        # Save validation results
        results_dir = Path('results')
        with open(results_dir / 'validation_metrics.json', 'w') as f:
            json.dump(validation_metrics, f, indent=4)
            
        # Generate validation plots
        self._plot_validation_results(solution, results_dir)
        
        return validation_metrics
        
    def _plot_validation_results(self, solution, results_dir: Path):
        """Generate detailed validation plots comparing simulation to experimental data"""
        sim_time = solution.getTimeMat()
        exp_time = self.experimental_data.time
        
        # Set up the plotting style
        plt.style.use('seaborn')
        
        # 1. Joint Angles Comparison
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        joints = ['hip', 'knee', 'ankle']
        
        for ax, joint in zip(axes, joints):
            sim_angle = np.rad2deg(solution.getStateMat(f'/jointset/{joint}/value'))
            exp_angle = self.experimental_data.joint_angles[joint]
            
            # Interpolate simulation data
            f = interpolate.interp1d(sim_time, sim_angle, bounds_error=False, fill_value="extrapolate")
            sim_angle_interp = f(exp_time)
            
            ax.plot(exp_time, sim_angle_interp, 'b-', label='Simulation')
            ax.plot(exp_time, exp_angle, 'r--', label='Experimental')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Angle (degrees)')
            ax.set_title(f'{joint.title()} Joint Angle')
            ax.legend()
            ax.grid(True)
            
        plt.tight_layout()
        plt.savefig(results_dir / 'validation_joint_angles.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Ground Reaction Forces Comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Vertical GRF
        sim_vgrf = solution.getStateMat('/forceset/contactHeel_r/value_y')
        exp_vgrf = self.experimental_data.ground_forces['vertical']
        
        f = interpolate.interp1d(sim_time, sim_vgrf, bounds_error=False, fill_value="extrapolate")
        sim_vgrf_interp = f(exp_time)
        
        ax1.plot(exp_time, sim_vgrf_interp, 'b-', label='Simulation')
        ax1.plot(exp_time, exp_vgrf, 'r--', label='Experimental')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Force (N)')
        ax1.set_title('Vertical Ground Reaction Force')
        ax1.legend()
        ax1.grid(True)
        
        # Horizontal GRF
        sim_hgrf = solution.getStateMat('/forceset/contactHeel_r/value_x')
        exp_hgrf = self.experimental_data.ground_forces['horizontal']
        
        f = interpolate.interp1d(sim_time, sim_hgrf, bounds_error=False, fill_value="extrapolate")
        sim_hgrf_interp = f(exp_time)
        
        ax2.plot(exp_time, sim_hgrf_interp, 'b-', label='Simulation')
        ax2.plot(exp_time, exp_hgrf, 'r--', label='Experimental')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Force (N)')
        ax2.set_title('Horizontal Ground Reaction Force')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'validation_ground_forces.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Muscle Activation vs EMG Comparison
        fig, axes = plt.subplots(3, 2, figsize=(15, 20))
        muscles = [
            ('hip_flexor_r', 'hip_flexor'),
            ('hip_extensor_r', 'hip_extensor'),
            ('knee_extensor_r', 'knee_extensor'),
            ('knee_flexor_r', 'knee_flexor'),
            ('ankle_plantar_r', 'ankle_plantar'),
            ('ankle_dorsi_r', 'ankle_dorsi')
        ]
        
        for (muscle, emg_key), ax in zip(muscles, axes.flat):
            sim_activation = solution.getControlMat(f'/forceset/{muscle}/activation')
            exp_emg = self.experimental_data.emg[emg_key]
            
            f = interpolate.interp1d(sim_time, sim_activation, bounds_error=False, fill_value="extrapolate")
            sim_activation_interp = f(exp_time)
            
            ax.plot(exp_time, sim_activation_interp, 'b-', label='Simulation')
            ax.plot(exp_time, exp_emg, 'r--', label='EMG')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Activation')
            ax.set_title(f'{muscle} Activation vs EMG')
            ax.legend()
            ax.grid(True)
            
        plt.tight_layout()
        plt.savefig(results_dir / 'validation_muscle_activation.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run the simulation"""
    sim = Walking2DSimulation()
    
    # Load experimental data if available
    exp_data_path = Path('data/experimental_gait_data.csv')
    if exp_data_path.exists():
        sim.load_experimental_data(exp_data_path)
    
    # Create and configure model
    model = sim.create_model()
    sim.setup_muscles()
    sim.setup_metabolics()
    
    # Run simulation
    study = sim.create_study()
    solution = sim.solve()
    
    # Analyze and validate results
    metrics = sim.analyze_results(solution)
    if sim.experimental_data is not None:
        validation_metrics = sim.validate_results(solution)
    
    # Visualize results
    sim.visualize(solution)
    
    # Export model for visualization in OpenSim Creator
    sim.export_model()

if __name__ == "__main__":
    main() 