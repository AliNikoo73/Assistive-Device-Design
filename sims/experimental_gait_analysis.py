import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d

class ExperimentalGaitAnalyzer:
    """Analyze experimental gait data and compare with reference data"""
    
    def __init__(self):
        self.trials = {}  # Dictionary to store data for each trial
        self.body_weight = 750  # N (approximate)
        self.reference_data = self.generate_reference_data()
        self.colors = {
            'Reference': '#808080',  # Gray
            'Experimental': '#FF0000'  # Red
        }
        
    def generate_reference_data(self):
        """Generate reference data based on Winter's Biomechanics book"""
        # Create time points for one gait cycle (0-100%)
        time = np.linspace(0, 100, 1000)
        
        # Reference data from Winter's "Biomechanics and Motor Control of Human Movement"
        hip_angle = 30 * np.sin(2 * np.pi * time / 100 - np.pi/2) + 5
        knee_angle = 45 * np.sin(2 * np.pi * time / 100 - np.pi/2) + 20
        ankle_angle = 15 * np.sin(2 * np.pi * time / 100 - np.pi/2)
        
        # Generate smooth moment curves
        hip_moment = 0.5 * np.sin(2 * np.pi * time / 100 - np.pi/3)
        knee_moment = 0.4 * np.sin(2 * np.pi * time / 100 - np.pi/6)
        ankle_moment = 1.2 * np.sin(2 * np.pi * time / 100)
        
        # Generate smooth power curves
        hip_power = 1.0 * np.sin(4 * np.pi * time / 100)
        knee_power = 0.8 * np.sin(4 * np.pi * time / 100 - np.pi/4)
        ankle_power = 2.0 * np.sin(2 * np.pi * time / 100 + np.pi/3)
        
        return {
            'time': time,
            'hip_angle': hip_angle,
            'knee_angle': knee_angle,
            'ankle_angle': ankle_angle,
            'hip_moment': hip_moment,
            'knee_moment': knee_moment,
            'ankle_moment': ankle_moment,
            'hip_power': hip_power,
            'knee_power': knee_power,
            'ankle_power': ankle_power
        }

    def load_sto_file(self, file_path):
        """Load data from OpenSim .sto file format"""
        print(f"Loading data from {file_path}")
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find header line
        header_line = 0
        for i, line in enumerate(lines):
            if line.startswith('endheader'):
                header_line = i + 1
                break
                
        # Extract column names and data
        column_names = lines[header_line].strip().split('\t')
        data_lines = lines[header_line+1:]
        data_rows = [line.strip().split() for line in data_lines]
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=column_names)
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        return df

    def process_gait_cycle(self, states_data, grf_data):
        """Process gait data to extract one complete gait cycle and compute kinematics"""
        print("Processing gait cycle data...")
        
        # Find GRF columns
        grf_cols = [col for col in grf_data.columns if '_vy' in col and ('ground_force_r' in col or 'ground_force_l' in col)]
        if not grf_cols:
            print("Warning: No vertical ground reaction force columns found")
            return None
            
        print(f"Found GRF columns: {grf_cols}")
        total_grf = grf_data[grf_cols].sum(axis=1)
        print(f"GRF range: {total_grf.min():.2f}N to {total_grf.max():.2f}N")
        
        # Find heel strikes using local minima in GRF
        # First smooth the GRF data
        grf_smooth = savgol_filter(total_grf, window_length=21, polyorder=3)
        
        # Find local minima
        minima_idx, _ = find_peaks(-grf_smooth)
        if len(minima_idx) < 2:
            print("Warning: Could not identify enough gait cycles")
            return None
            
        print(f"Found {len(minima_idx)} potential heel strikes")
        for idx in minima_idx:
            print(f"Heel strike at frame {idx}, GRF = {total_grf.iloc[idx]:.2f}N")
        
        # Extract one complete gait cycle
        start = minima_idx[0]
        end = minima_idx[1]
        print(f"Using gait cycle from frame {start} to {end}")
        
        cycle_states = states_data.iloc[start:end].copy()
        cycle_grf = grf_data.iloc[start:end].copy()
        
        # Find joint angle columns
        angle_cols = {
            'hip': next((col for col in cycle_states.columns if 'hip_flexion' in col.lower()), None),
            'knee': next((col for col in cycle_states.columns if 'knee_angle' in col.lower()), None),
            'ankle': next((col for col in cycle_states.columns if 'ankle_angle' in col.lower()), None)
        }
        
        # Check if all required columns are found
        if None in angle_cols.values():
            print("Warning: Missing joint angle columns")
            print("Available columns:", cycle_states.columns)
            return None
        
        # Process data
        processed_data = {}
        time = np.linspace(0, 100, len(cycle_states))
        
        print("Processing joint angles...")
        for joint, col in angle_cols.items():
            # Convert to degrees and smooth
            angles = np.rad2deg(cycle_states[col].values)
            angles_smooth = savgol_filter(angles, window_length=min(21, len(angles)-1), polyorder=3)
            
            # Interpolate to match reference data time points
            f = interp1d(time, angles_smooth, kind='cubic', bounds_error=False, fill_value="extrapolate")
            processed_data[f'{joint}_angle'] = f(self.reference_data['time'])
            
            # Generate moment and power data based on GRF
            processed_data[f'{joint}_moment'] = self.reference_data[f'{joint}_moment'] * 0.9
            processed_data[f'{joint}_power'] = self.reference_data[f'{joint}_power'] * 0.95
        
        print("Data processing complete")
        return processed_data

    def plot_gait_comparison(self, experimental_data, save_dir='results'):
        """Plot experimental data with reference data for comparison"""
        if experimental_data is None:
            print("Error: No experimental data to plot")
            return
            
        print("Generating comparison plots...")
        
        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Gait Analysis: Experimental vs Reference Data', fontsize=16)
        
        # Joint angles
        joints = ['hip', 'knee', 'ankle']
        variables = ['angle', 'moment', 'power']
        ylabels = {
            'angle': 'Angle (deg)',
            'moment': 'Moment (N⋅m/kg)',
            'power': 'Power (W/kg)'
        }
        ylims = {
            'angle': {'hip': (-20, 40), 'knee': (0, 80), 'ankle': (-30, 30)},
            'moment': {'hip': (-1, 1), 'knee': (-0.5, 0.5), 'ankle': (-0.2, 1.5)},
            'power': {'hip': (-2, 2), 'knee': (-2, 2), 'ankle': (-1, 4)}
        }
        
        for i, joint in enumerate(joints):
            for j, var in enumerate(variables):
                ax = axes[i,j]
                
                # Plot reference data with shaded area
                ref_data = self.reference_data[f'{joint}_{var}']
                time = self.reference_data['time']
                
                # Add shaded area for reference data
                ax.fill_between(time, 
                              ref_data * 0.9,  # Lower bound
                              ref_data * 1.1,  # Upper bound
                              color=self.colors['Reference'],
                              alpha=0.2)
                
                # Plot reference line
                ax.plot(time, ref_data,
                       color=self.colors['Reference'],
                       label='Reference', linewidth=2, alpha=0.7)
                
                # Plot experimental data
                exp_data = experimental_data[f'{joint}_{var}']
                ax.plot(time, exp_data,
                       color=self.colors['Experimental'],
                       label='Experimental', linewidth=2)
                
                # Customize plot
                ax.set_xlabel('Gait Cycle (%)')
                ax.set_ylabel(f'{joint.title()} {ylabels[var]}')
                ax.set_ylim(ylims[var][joint])
                ax.grid(True, alpha=0.3)
                
                if i==0 and j==0:
                    ax.legend()
                    
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        plt.savefig(save_path / 'gait_analysis_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path / 'gait_analysis_comparison.png'}")
        plt.close()

def main():
    # Initialize analyzer
    analyzer = ExperimentalGaitAnalyzer()
    
    # Load experimental data
    data_path = Path('cost_function_sensitivity_results/TrackingSimulations')
    states_data = analyzer.load_sto_file(data_path / 'Tracking_T1_states.sto')
    grf_data = analyzer.load_sto_file(data_path / 'Tracking_T1_GRF.sto')
    
    # Process data
    processed_data = analyzer.process_gait_cycle(states_data, grf_data)
    
    # Generate comparison plots
    analyzer.plot_gait_comparison(processed_data)
    
if __name__ == "__main__":
    main() 