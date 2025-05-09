import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import interpolate

from synthetic_gait import SyntheticGaitGenerator
from experimental_gait_analysis import ExperimentalGaitAnalyzer

class GaitAnalysisComparator:
    """Compare experimental and synthetic gait data"""
    
    def __init__(self):
        self.exp_data = None
        self.synthetic_data = None
        self.exp_time = None
        self.synth_time = None
        
    def load_experimental_data(self):
        """Load experimental data from tracking files"""
        analyzer = ExperimentalGaitAnalyzer()
        
        # Load GRF and state data
        grf_file = Path('cost_function_sensitivity_results/TrackingSimulations/Tracking_T1_GRF.sto')
        state_file = Path('cost_function_sensitivity_results/TrackingSimulations/Tracking_T1_states.sto')
        
        if grf_file.exists():
            analyzer.load_data(grf_file, state_file if state_file.exists() else None)
            
            # Extract data
            exp_data = {
                'time': analyzer.time
            }
            
            # Extract GRF data
            grfs = analyzer.extract_ground_reaction_forces()
            if grfs:
                for key, value in grfs.items():
                    exp_data[f'grf_{key}'] = value
            
            # Extract joint angles if state data is available
            if analyzer.state_data is not None:
                joint_angles = analyzer.extract_joint_angles()
                if joint_angles:
                    for joint, angle in joint_angles.items():
                        exp_data[f'{joint}_angle'] = angle
            
            self.exp_data = exp_data
            self.exp_time = exp_data['time']
            return exp_data
        else:
            print(f"Error: GRF file not found at {grf_file}")
            return None
            
    def generate_synthetic_data(self):
        """Generate synthetic gait data"""
        # Create synthetic data generator with matching cycle duration
        if self.exp_data is None:
            # Use default duration if no experimental data
            cycle_duration = 1.0
        else:
            # Match experimental data duration
            cycle_duration = self.exp_data['time'][-1] - self.exp_data['time'][0]
            
        generator = SyntheticGaitGenerator(cycle_duration=cycle_duration, sampling_rate=100)
        synthetic_data = generator.generate_complete_gait_dataset()
        
        self.synthetic_data = synthetic_data
        self.synth_time = synthetic_data['time'].values
        return synthetic_data
        
    def compare_data(self):
        """Compare experimental and synthetic data"""
        if self.exp_data is None or self.synthetic_data is None:
            print("Error: Both experimental and synthetic data must be loaded first")
            return
            
        # Resample synthetic data to match experimental time points for direct comparison
        comparison = {
            'time': self.exp_time
        }
        
        # Function to interpolate synthetic data to match experimental time points
        def interpolate_synth_data(data_col):
            f = interpolate.interp1d(self.synth_time, data_col, 
                                     bounds_error=False, fill_value="extrapolate")
            return f(self.exp_time)
            
        # Compare joint angles
        for joint in ['hip', 'knee', 'ankle']:
            exp_key = f'{joint}_angle'
            synth_key = f'{joint}_angle'
            
            if exp_key in self.exp_data and synth_key in self.synthetic_data:
                # Experimental data is already in the right time basis
                comparison[f'exp_{exp_key}'] = self.exp_data[exp_key]
                
                # Resample synthetic data
                comparison[f'synth_{synth_key}'] = interpolate_synth_data(self.synthetic_data[synth_key])
                
        # Compare ground reaction forces
        for force_type in ['vertical', 'anterior_posterior']:
            exp_key = f'grf_{force_type}'
            synth_key = f'grf_{force_type}'
            
            if exp_key in self.exp_data and synth_key in self.synthetic_data:
                # Experimental data is already in the right time basis
                comparison[f'exp_{exp_key}'] = self.exp_data[exp_key]
                
                # Resample synthetic data
                comparison[f'synth_{synth_key}'] = interpolate_synth_data(self.synthetic_data[synth_key])
                
        return comparison
        
    def normalize_gait_cycle(self, data, time_col='time'):
        """Normalize time to gait cycle percentage (0-100%)"""
        if data is None:
            return None
            
        # Create a copy of the data
        normalized = data.copy()
        
        # Calculate cycle duration
        start_time = data[time_col].iloc[0]
        end_time = data[time_col].iloc[-1]
        duration = end_time - start_time
        
        # Normalize time to percentage (0-100%)
        normalized[time_col] = ((data[time_col] - start_time) / duration) * 100.0
        
        return normalized
        
    def plot_comparison(self, comparison=None, normalize=True, show_plots=True, save_plots=True):
        """Plot comparison between experimental and synthetic data
        
        Args:
            comparison: Dictionary containing comparison data
            normalize: Whether to normalize time to gait cycle percentage
            show_plots: Whether to display plots
            save_plots: Whether to save plots to files
        """
        if comparison is None:
            comparison = self.compare_data()
            
        if comparison is None:
            print("Error: No comparison data available")
            return
            
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        # Set up plot style
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # Create figures list to return
        figures = []
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame(comparison)
        
        # Normalize to gait cycle percentage if requested
        if normalize:
            df = self.normalize_gait_cycle(df)
            x_label = 'Gait Cycle (%)'
        else:
            x_label = 'Time (s)'
            
        # 1. Joint Angle Comparison
        joint_cols = {'hip': 'r', 'knee': 'g', 'ankle': 'b'}
        fig_angles, axes_angles = plt.subplots(len(joint_cols), 1, figsize=(10, 12), sharex=True)
        
        for i, (joint, color) in enumerate(joint_cols.items()):
            exp_key = f'exp_{joint}_angle'
            synth_key = f'synth_{joint}_angle'
            
            if exp_key in df.columns and synth_key in df.columns:
                axes_angles[i].plot(df['time'], df[exp_key], f'{color}-', linewidth=2, label='Experimental')
                axes_angles[i].plot(df['time'], df[synth_key], f'{color}--', linewidth=2, label='Synthetic')
                axes_angles[i].set_ylabel('Angle (rad)')
                axes_angles[i].set_title(f'{joint.capitalize()} Joint Angle')
                axes_angles[i].legend()
                axes_angles[i].grid(True)
                
        axes_angles[-1].set_xlabel(x_label)
        plt.tight_layout()
        
        if save_plots:
            fig_angles.savefig(output_dir / 'comparison_joint_angles.png', dpi=300, bbox_inches='tight')
            
        figures.append(fig_angles)
        
        # 2. Ground Reaction Force Comparison
        grf_cols = {'vertical': 'k', 'anterior_posterior': 'b'}
        fig_grf, axes_grf = plt.subplots(len(grf_cols), 1, figsize=(10, 8), sharex=True)
        
        for i, (force_type, color) in enumerate(grf_cols.items()):
            exp_key = f'exp_grf_{force_type}'
            synth_key = f'synth_grf_{force_type}'
            
            if exp_key in df.columns and synth_key in df.columns:
                axes_grf[i].plot(df['time'], df[exp_key], f'{color}-', linewidth=2, label='Experimental')
                axes_grf[i].plot(df['time'], df[synth_key], f'{color}--', linewidth=2, label='Synthetic')
                axes_grf[i].set_ylabel('Force (N)')
                axes_grf[i].set_title(f'{force_type.capitalize()} Ground Reaction Force')
                axes_grf[i].legend()
                axes_grf[i].grid(True)
                
        axes_grf[-1].set_xlabel(x_label)
        plt.tight_layout()
        
        if save_plots:
            fig_grf.savefig(output_dir / 'comparison_ground_forces.png', dpi=300, bbox_inches='tight')
            
        figures.append(fig_grf)
        
        # 3. Combined visualization
        fig_combined = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])
        
        # Joint angles - Hip
        ax1 = fig_combined.add_subplot(gs[0])
        if 'exp_hip_angle' in df.columns and 'synth_hip_angle' in df.columns:
            ax1.plot(df['time'], df['exp_hip_angle'], 'r-', linewidth=2, label='Exp Hip')
            ax1.plot(df['time'], df['synth_hip_angle'], 'r--', linewidth=2, label='Synth Hip')
            ax1.set_ylabel('Angle (rad)')
            ax1.set_title('Hip Joint Angle')
            ax1.legend(loc='upper right')
            ax1.grid(True)
            
        # Joint angles - Knee
        ax2 = fig_combined.add_subplot(gs[1], sharex=ax1)
        if 'exp_knee_angle' in df.columns and 'synth_knee_angle' in df.columns:
            ax2.plot(df['time'], df['exp_knee_angle'], 'g-', linewidth=2, label='Exp Knee')
            ax2.plot(df['time'], df['synth_knee_angle'], 'g--', linewidth=2, label='Synth Knee')
            ax2.set_ylabel('Angle (rad)')
            ax2.set_title('Knee Joint Angle')
            ax2.legend(loc='upper right')
            ax2.grid(True)
            
        # Ground reaction forces - Vertical
        ax3 = fig_combined.add_subplot(gs[2], sharex=ax1)
        if 'exp_grf_vertical' in df.columns and 'synth_grf_vertical' in df.columns:
            ax3.plot(df['time'], df['exp_grf_vertical'], 'k-', linewidth=2, label='Exp Vertical GRF')
            ax3.plot(df['time'], df['synth_grf_vertical'], 'k--', linewidth=2, label='Synth Vertical GRF')
            ax3.set_ylabel('Force (N)')
            ax3.set_title('Vertical Ground Reaction Force')
            ax3.legend(loc='upper right')
            ax3.grid(True)
            
        # Ground reaction forces - Horizontal
        ax4 = fig_combined.add_subplot(gs[3], sharex=ax1)
        if 'exp_grf_anterior_posterior' in df.columns and 'synth_grf_anterior_posterior' in df.columns:
            ax4.plot(df['time'], df['exp_grf_anterior_posterior'], 'b-', linewidth=2, label='Exp A-P GRF')
            ax4.plot(df['time'], df['synth_grf_anterior_posterior'], 'b--', linewidth=2, label='Synth A-P GRF')
            ax4.set_xlabel(x_label)
            ax4.set_ylabel('Force (N)')
            ax4.set_title('Anterior-Posterior Ground Reaction Force')
            ax4.legend(loc='upper right')
            ax4.grid(True)
            
        plt.tight_layout()
        
        if save_plots:
            fig_combined.savefig(output_dir / 'comparison_gait_analysis.png', dpi=300, bbox_inches='tight')
            
        figures.append(fig_combined)
        
        if show_plots:
            plt.show()
        else:
            plt.close('all')
            
        return figures
        
def main():
    # Initialize the comparator
    print("Initializing gait analysis comparator...")
    comparator = GaitAnalysisComparator()
    
    # Load experimental data
    print("Loading experimental data...")
    comparator.load_experimental_data()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    comparator.generate_synthetic_data()
    
    # Compare the data
    print("Comparing experimental and synthetic data...")
    comparison = comparator.compare_data()
    
    # Plot comparison
    print("Generating comparison plots...")
    figures = comparator.plot_comparison(comparison)
    
    print("Comparison complete! Results saved to the 'results' directory.")
    return figures
    
if __name__ == "__main__":
    main() 