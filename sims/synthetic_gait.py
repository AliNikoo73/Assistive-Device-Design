import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy import signal

class SyntheticGaitGenerator:
    """Generate synthetic gait data for visualization and testing"""
    
    def __init__(self, cycle_duration=1.0, sampling_rate=100):
        """Initialize the gait generator
        
        Args:
            cycle_duration: Duration of gait cycle in seconds
            sampling_rate: Number of samples per second
        """
        self.cycle_duration = cycle_duration
        self.sampling_rate = sampling_rate
        self.num_samples = int(cycle_duration * sampling_rate)
        self.time = np.linspace(0, cycle_duration, self.num_samples)
        
    def generate_joint_angles(self):
        """Generate synthetic joint angles for a gait cycle
        
        Returns:
            Dictionary of joint angles (hip, knee, ankle)
        """
        # Hip angle: sinusoidal with appropriate phase
        hip_angle = 10 * np.sin(2 * np.pi * self.time / self.cycle_duration) + 5
        
        # Knee angle: more complex pattern with flexion during swing
        t_norm = self.time / self.cycle_duration
        knee_angle = np.zeros_like(t_norm)
        
        # Stance phase (0-60%)
        stance_mask = t_norm < 0.6
        knee_angle[stance_mask] = 15 * np.sin(np.pi * t_norm[stance_mask] / 0.6)
        
        # Swing phase (60-100%)
        swing_mask = t_norm >= 0.6
        knee_angle[swing_mask] = 60 * np.sin(np.pi * (t_norm[swing_mask] - 0.6) / 0.4)
        
        # Ankle angle: dorsiflexion during stance, plantarflexion during push-off
        ankle_angle = np.zeros_like(t_norm)
        ankle_angle[t_norm < 0.5] = -5 * np.sin(2 * np.pi * t_norm[t_norm < 0.5])
        ankle_angle[t_norm >= 0.5] = -20 * np.sin(np.pi * (t_norm[t_norm >= 0.5] - 0.5) / 0.5)
        
        return {
            'hip': hip_angle,
            'knee': knee_angle,
            'ankle': ankle_angle
        }
        
    def generate_ground_reaction_forces(self):
        """Generate synthetic ground reaction forces for a gait cycle
        
        Returns:
            Dictionary of GRFs (vertical, anterior-posterior)
        """
        t_norm = self.time / self.cycle_duration
        
        # Vertical GRF: double-peak pattern
        vertical_grf = np.zeros_like(t_norm)
        stance_mask = t_norm < 0.6
        vertical_grf[stance_mask] = 800 * (
            0.8 * np.exp(-((t_norm[stance_mask] - 0.15) / 0.1)**2) + 
            1.0 * np.exp(-((t_norm[stance_mask] - 0.45) / 0.1)**2)
        )
        
        # Anterior-posterior GRF: braking followed by propulsion
        ap_grf = np.zeros_like(t_norm)
        ap_grf[stance_mask] = -150 * np.sin(2 * np.pi * t_norm[stance_mask] / 0.6)
        
        return {
            'vertical': vertical_grf,
            'anterior_posterior': ap_grf
        }
        
    def generate_muscle_activations(self):
        """Generate synthetic muscle activations for a gait cycle
        
        Returns:
            Dictionary of muscle activations for major muscle groups
        """
        t_norm = self.time / self.cycle_duration
        
        # Create a base function for generating activation patterns
        def create_activation(peak_time, duration, amplitude=1.0):
            return amplitude * np.exp(-((t_norm - peak_time) / (duration / 2.5))**2)
        
        # Hip flexor (active during late stance and swing)
        hip_flexor = create_activation(0.7, 0.3, 0.8)
        
        # Hip extensor (active during early stance)
        hip_extensor = create_activation(0.1, 0.2, 0.9)
        
        # Knee extensor (quadriceps, active during early stance)
        knee_extensor = create_activation(0.15, 0.2, 0.7) + 0.3 * create_activation(0.7, 0.15, 0.5)
        
        # Knee flexor (hamstrings, active during late swing)
        knee_flexor = create_activation(0.9, 0.2, 0.6) + 0.4 * create_activation(0.3, 0.2, 0.3)
        
        # Ankle plantarflexor (gastrocnemius, active during late stance)
        ankle_plantar = create_activation(0.4, 0.3, 1.0)
        
        # Ankle dorsiflexor (tibialis anterior, active during swing)
        ankle_dorsi = create_activation(0.8, 0.4, 0.7) + 0.3 * create_activation(0.05, 0.1, 0.6)
        
        # Add noise to make it more realistic
        def add_noise(signal, noise_level=0.05):
            noise = noise_level * np.random.randn(len(signal))
            noisy_signal = signal + noise
            return np.clip(noisy_signal, 0, 1)  # Constrain between 0 and 1
            
        return {
            'hip_flexor': add_noise(hip_flexor),
            'hip_extensor': add_noise(hip_extensor),
            'knee_extensor': add_noise(knee_extensor),
            'knee_flexor': add_noise(knee_flexor),
            'ankle_plantar': add_noise(ankle_plantar),
            'ankle_dorsi': add_noise(ankle_dorsi)
        }
        
    def generate_complete_gait_dataset(self):
        """Generate a complete dataset for a gait cycle
        
        Returns:
            DataFrame containing all gait data
        """
        joint_angles = self.generate_joint_angles()
        grfs = self.generate_ground_reaction_forces()
        emg = self.generate_muscle_activations()
        
        data = {'time': self.time}
        
        # Add joint angles
        for joint, angles in joint_angles.items():
            data[f'{joint}_angle'] = angles
            
        # Add ground reaction forces
        for force_type, force in grfs.items():
            data[f'grf_{force_type}'] = force
            
        # Add muscle activations
        for muscle, activation in emg.items():
            data[f'{muscle}_activation'] = activation
            
        return pd.DataFrame(data)
        
    def save_data(self, data, filename='synthetic_gait_data.csv'):
        """Save the generated data to a CSV file
        
        Args:
            data: DataFrame containing gait data
            filename: Name of file to save data to
        """
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        data.to_csv(output_dir / filename, index=False)
        print(f"Data saved to {output_dir / filename}")
        
    def plot_results(self, data, show_plots=True, save_plots=True):
        """Plot the gait cycle data
        
        Args:
            data: DataFrame containing gait data
            show_plots: Whether to display plots
            save_plots: Whether to save plots to files
        """
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        # Set up plot style
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # 1. Joint Angles
        fig_angles, ax_angles = plt.subplots(figsize=(10, 6))
        ax_angles.plot(data['time'], data['hip_angle'], 'r-', linewidth=2, label='Hip')
        ax_angles.plot(data['time'], data['knee_angle'], 'g-', linewidth=2, label='Knee')
        ax_angles.plot(data['time'], data['ankle_angle'], 'b-', linewidth=2, label='Ankle')
        ax_angles.set_xlabel('Time (s)')
        ax_angles.set_ylabel('Angle (degrees)')
        ax_angles.set_title('Joint Angles During Gait Cycle')
        ax_angles.legend()
        ax_angles.grid(True)
        
        if save_plots:
            fig_angles.savefig(output_dir / 'joint_angles.png', dpi=300, bbox_inches='tight')
            
        # 2. Ground Reaction Forces
        fig_grf, ax_grf = plt.subplots(figsize=(10, 6))
        ax_grf.plot(data['time'], data['grf_vertical'], 'k-', linewidth=2, label='Vertical')
        ax_grf.plot(data['time'], data['grf_anterior_posterior'], 'k--', linewidth=2, label='Ant-Post')
        ax_grf.set_xlabel('Time (s)')
        ax_grf.set_ylabel('Force (N)')
        ax_grf.set_title('Ground Reaction Forces During Gait Cycle')
        ax_grf.legend()
        ax_grf.grid(True)
        
        if save_plots:
            fig_grf.savefig(output_dir / 'ground_reaction_forces.png', dpi=300, bbox_inches='tight')
            
        # 3. Muscle Activations
        fig_emg, axes_emg = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        axes_emg = axes_emg.flatten()
        
        muscles = [
            ('hip_flexor_activation', 'Hip Flexor'),
            ('hip_extensor_activation', 'Hip Extensor'),
            ('knee_extensor_activation', 'Knee Extensor'),
            ('knee_flexor_activation', 'Knee Flexor'),
            ('ankle_plantar_activation', 'Ankle Plantarflexor'),
            ('ankle_dorsi_activation', 'Ankle Dorsiflexor')
        ]
        
        for i, (col, title) in enumerate(muscles):
            axes_emg[i].plot(data['time'], data[col], linewidth=2)
            axes_emg[i].set_ylabel('Activation')
            axes_emg[i].set_title(title)
            axes_emg[i].grid(True)
            
        axes_emg[-1].set_xlabel('Time (s)')
        fig_emg.suptitle('Muscle Activations During Gait Cycle', fontsize=16)
        fig_emg.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_plots:
            fig_emg.savefig(output_dir / 'muscle_activations.png', dpi=300, bbox_inches='tight')
        
        # 4. Combined visualization
        fig_combined = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1])
        
        # Joint angles
        ax1 = fig_combined.add_subplot(gs[0, :])
        ax1.plot(data['time'], data['hip_angle'], 'r-', linewidth=2, label='Hip')
        ax1.plot(data['time'], data['knee_angle'], 'g-', linewidth=2, label='Knee')
        ax1.plot(data['time'], data['ankle_angle'], 'b-', linewidth=2, label='Ankle')
        ax1.set_ylabel('Angle (degrees)')
        ax1.set_title('Joint Angles')
        ax1.legend(loc='upper right')
        ax1.grid(True)
        
        # Ground reaction forces
        ax2 = fig_combined.add_subplot(gs[1, :])
        ax2.plot(data['time'], data['grf_vertical'], 'k-', linewidth=2, label='Vertical')
        ax2.plot(data['time'], data['grf_anterior_posterior'], 'k--', linewidth=2, label='Ant-Post')
        ax2.set_ylabel('Force (N)')
        ax2.set_title('Ground Reaction Forces')
        ax2.legend(loc='upper right')
        ax2.grid(True)
        
        # Hip muscles
        ax3 = fig_combined.add_subplot(gs[2, 0])
        ax3.plot(data['time'], data['hip_flexor_activation'], 'r-', linewidth=2, label='Flexor')
        ax3.plot(data['time'], data['hip_extensor_activation'], 'b-', linewidth=2, label='Extensor')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Activation')
        ax3.set_title('Hip Muscle Activations')
        ax3.legend(loc='upper right')
        ax3.grid(True)
        
        # Ankle muscles
        ax4 = fig_combined.add_subplot(gs[2, 1])
        ax4.plot(data['time'], data['ankle_plantar_activation'], 'r-', linewidth=2, label='Plantarflexor')
        ax4.plot(data['time'], data['ankle_dorsi_activation'], 'b-', linewidth=2, label='Dorsiflexor')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Activation')
        ax4.set_title('Ankle Muscle Activations')
        ax4.legend(loc='upper right')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            fig_combined.savefig(output_dir / 'combined_gait_analysis.png', dpi=300, bbox_inches='tight')
            
        if show_plots:
            plt.show()
        else:
            plt.close('all')
            
        return [fig_angles, fig_grf, fig_emg, fig_combined]

def main():
    # Generate synthetic gait data for a complete cycle
    print("Generating synthetic gait data...")
    generator = SyntheticGaitGenerator(cycle_duration=1.2, sampling_rate=100)
    gait_data = generator.generate_complete_gait_dataset()
    
    # Save data to CSV
    generator.save_data(gait_data)
    
    # Plot and save results
    print("Plotting results...")
    figures = generator.plot_results(gait_data, show_plots=True)
    
    print("Done! Results saved to the 'results' directory.")
    return figures
    
if __name__ == "__main__":
    main() 