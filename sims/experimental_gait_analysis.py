import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

class ExperimentalGaitAnalyzer:
    """Analyze experimental gait data from OpenSim tracking files"""
    
    def __init__(self):
        self.grf_data = None
        self.state_data = None
        self.time = None
        
    def load_sto_file(self, file_path):
        """Load data from OpenSim .sto file format
        
        Args:
            file_path: Path to .sto file
            
        Returns:
            DataFrame containing the data
        """
        # Read the file contents
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the header line
        header_line = 0
        for i, line in enumerate(lines):
            if line.startswith('endheader'):
                header_line = i + 1
                break
        
        # Extract column names
        column_names = lines[header_line].strip().split('\t')
        
        # Parse data
        data_lines = lines[header_line+1:]
        data_rows = [line.strip().split() for line in data_lines]
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=column_names)
        
        # Convert data types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        return df
    
    def load_data(self, grf_file_path, state_file_path=None):
        """Load experimental data from GRF and state files
        
        Args:
            grf_file_path: Path to ground reaction force .sto file
            state_file_path: Path to states .sto file (optional)
        """
        print(f"Loading GRF data from {grf_file_path}")
        self.grf_data = self.load_sto_file(grf_file_path)
        self.time = self.grf_data['time'].values
        
        if state_file_path:
            print(f"Loading state data from {state_file_path}")
            self.state_data = self.load_sto_file(state_file_path)
    
    def extract_joint_angles(self):
        """Extract joint angles from state data
        
        Returns:
            Dictionary containing joint angles for hip, knee, and ankle
        """
        if self.state_data is None:
            raise ValueError("No state data loaded")
        
        # Find columns for joint angles
        # Typical naming: hip_flexion_r, knee_angle_r, ankle_angle_r
        joint_angles = {}
        
        # Extract hip angle
        hip_col = [col for col in self.state_data.columns if 'hip' in col.lower() and 'value' in col.lower()]
        if hip_col:
            joint_angles['hip'] = self.state_data[hip_col[0]].values
        
        # Extract knee angle
        knee_col = [col for col in self.state_data.columns if 'knee' in col.lower() and 'value' in col.lower()]
        if knee_col:
            joint_angles['knee'] = self.state_data[knee_col[0]].values
        
        # Extract ankle angle
        ankle_col = [col for col in self.state_data.columns if 'ankle' in col.lower() and 'value' in col.lower()]
        if ankle_col:
            joint_angles['ankle'] = self.state_data[ankle_col[0]].values
            
        return joint_angles
    
    def extract_ground_reaction_forces(self):
        """Extract ground reaction forces
        
        Returns:
            Dictionary containing vertical and horizontal GRFs
        """
        if self.grf_data is None:
            raise ValueError("No GRF data loaded")
        
        # Extract vertical GRF (y component)
        v_grf_cols = [col for col in self.grf_data.columns if 'vy' in col.lower()]
        
        # Extract horizontal GRF (x component)
        h_grf_cols = [col for col in self.grf_data.columns if 'vx' in col.lower()]
        
        grfs = {}
        
        if v_grf_cols:
            # Sum forces from both feet for a complete picture
            grfs['vertical'] = self.grf_data[v_grf_cols].sum(axis=1).values
            
        if h_grf_cols:
            grfs['anterior_posterior'] = self.grf_data[h_grf_cols].sum(axis=1).values
            
        return grfs
        
    def plot_results(self, joint_angles=None, grfs=None, show_plots=True, save_plots=True):
        """Plot experimental gait data
        
        Args:
            joint_angles: Dictionary containing joint angle data
            grfs: Dictionary containing ground reaction force data
            show_plots: Whether to display plots
            save_plots: Whether to save plots to files
        """
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        # Set up plot style
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # Create figures list to return
        figures = []
        
        # 1. Joint Angles (if available)
        if joint_angles:
            fig_angles, ax_angles = plt.subplots(figsize=(10, 6))
            
            if 'hip' in joint_angles:
                ax_angles.plot(self.time, joint_angles['hip'], 'r-', linewidth=2, label='Hip')
            if 'knee' in joint_angles:
                ax_angles.plot(self.time, joint_angles['knee'], 'g-', linewidth=2, label='Knee')
            if 'ankle' in joint_angles:
                ax_angles.plot(self.time, joint_angles['ankle'], 'b-', linewidth=2, label='Ankle')
                
            ax_angles.set_xlabel('Time (s)')
            ax_angles.set_ylabel('Angle (rad)')
            ax_angles.set_title('Joint Angles During Gait Cycle')
            ax_angles.legend()
            ax_angles.grid(True)
            
            if save_plots:
                fig_angles.savefig(output_dir / 'experimental_joint_angles.png', dpi=300, bbox_inches='tight')
                
            figures.append(fig_angles)
            
        # 2. Ground Reaction Forces
        if grfs:
            fig_grf, ax_grf = plt.subplots(figsize=(10, 6))
            
            if 'vertical' in grfs:
                ax_grf.plot(self.time, grfs['vertical'], 'k-', linewidth=2, label='Vertical')
            if 'anterior_posterior' in grfs:
                ax_grf.plot(self.time, grfs['anterior_posterior'], 'k--', linewidth=2, label='Ant-Post')
                
            ax_grf.set_xlabel('Time (s)')
            ax_grf.set_ylabel('Force (N)')
            ax_grf.set_title('Ground Reaction Forces During Gait Cycle')
            ax_grf.legend()
            ax_grf.grid(True)
            
            if save_plots:
                fig_grf.savefig(output_dir / 'experimental_ground_forces.png', dpi=300, bbox_inches='tight')
                
            figures.append(fig_grf)
            
        # 3. Combined visualization
        if joint_angles and grfs:
            fig_combined = plt.figure(figsize=(15, 10))
            gs = plt.GridSpec(2, 1, height_ratios=[1, 1])
            
            # Joint angles
            ax1 = fig_combined.add_subplot(gs[0])
            if 'hip' in joint_angles:
                ax1.plot(self.time, joint_angles['hip'], 'r-', linewidth=2, label='Hip')
            if 'knee' in joint_angles:
                ax1.plot(self.time, joint_angles['knee'], 'g-', linewidth=2, label='Knee')
            if 'ankle' in joint_angles:
                ax1.plot(self.time, joint_angles['ankle'], 'b-', linewidth=2, label='Ankle')
            ax1.set_ylabel('Angle (rad)')
            ax1.set_title('Joint Angles')
            ax1.legend(loc='upper right')
            ax1.grid(True)
            
            # Ground reaction forces
            ax2 = fig_combined.add_subplot(gs[1], sharex=ax1)
            if 'vertical' in grfs:
                ax2.plot(self.time, grfs['vertical'], 'k-', linewidth=2, label='Vertical')
            if 'anterior_posterior' in grfs:
                ax2.plot(self.time, grfs['anterior_posterior'], 'k--', linewidth=2, label='Ant-Post')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Force (N)')
            ax2.set_title('Ground Reaction Forces')
            ax2.legend(loc='upper right')
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_plots:
                fig_combined.savefig(output_dir / 'experimental_gait_analysis.png', dpi=300, bbox_inches='tight')
                
            figures.append(fig_combined)
        
        if show_plots:
            plt.show()
        else:
            plt.close('all')
            
        return figures

def main():
    # Initialize the analyzer
    print("Initializing gait analyzer...")
    analyzer = ExperimentalGaitAnalyzer()
    
    # Load experimental data
    grf_file = Path('cost_function_sensitivity_results/TrackingSimulations/Tracking_T1_GRF.sto')
    state_file = Path('cost_function_sensitivity_results/TrackingSimulations/Tracking_T1_states.sto')
    
    if grf_file.exists():
        analyzer.load_data(grf_file, state_file if state_file.exists() else None)
        
        # Extract GRF data
        print("Extracting ground reaction forces...")
        grfs = analyzer.extract_ground_reaction_forces()
        
        # Extract joint angles if state data is available
        joint_angles = None
        if analyzer.state_data is not None:
            print("Extracting joint angles...")
            joint_angles = analyzer.extract_joint_angles()
            
        # Plot results
        print("Generating plots...")
        figures = analyzer.plot_results(joint_angles, grfs)
        
        print("Analysis complete! Results saved to the 'results' directory.")
        return figures
    else:
        print(f"Error: GRF file not found at {grf_file}")
        return None

if __name__ == "__main__":
    main() 