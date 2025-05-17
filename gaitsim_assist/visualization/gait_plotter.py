"""
GaitPlotter: Main class for visualizing gait simulation results.

This class provides methods for creating various plots of gait simulation results,
including joint angles, ground reaction forces, muscle activations, and more.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd

from ..simulation import SimulationResults


class GaitPlotter:
    """Class for visualizing gait simulation results."""
    
    def __init__(self, style: str = 'whitegrid', figsize: Tuple[int, int] = (12, 8)):
        """Initialize the gait plotter.
        
        Args:
            style: Seaborn style for plots
            figsize: Default figure size (width, height) in inches
        """
        self.style = style
        self.figsize = figsize
        sns.set_style(style)
    
    def plot_joint_angles(self, results: SimulationResults, 
                        joints: Optional[List[str]] = None,
                        normalize_gait_cycle: bool = True,
                        show_reference: bool = False,
                        reference_data: Optional[Dict[str, np.ndarray]] = None,
                        save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Plot joint angles over time or gait cycle.
        
        Args:
            results: Simulation results
            joints: List of joints to plot. If None, plots hip, knee, and ankle
            normalize_gait_cycle: Whether to normalize time to gait cycle (0-100%)
            show_reference: Whether to show reference data
            reference_data: Reference joint angle data
            save_path: Path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        # Default joints to plot
        if joints is None:
            joints = ['hip', 'knee', 'ankle']
        
        # Create figure
        fig, axes = plt.subplots(len(joints), 1, figsize=self.figsize, sharex=True)
        if len(joints) == 1:
            axes = [axes]
        
        # Get time or gait cycle percentage
        if normalize_gait_cycle:
            x = np.linspace(0, 100, len(results.time))
            xlabel = 'Gait Cycle (%)'
        else:
            x = results.time
            xlabel = 'Time (s)'
        
        # Plot each joint
        for i, joint in enumerate(joints):
            if joint in results.joint_angles:
                axes[i].plot(x, results.joint_angles[joint], 'b-', linewidth=2, 
                           label='Simulation')
                
                # Plot reference data if available
                if show_reference and reference_data is not None:
                    if joint in reference_data:
                        # Resample reference data to match simulation time points
                        if normalize_gait_cycle:
                            ref_x = np.linspace(0, 100, len(reference_data[joint]))
                        else:
                            ref_x = np.linspace(results.time[0], results.time[-1], 
                                              len(reference_data[joint]))
                        
                        axes[i].plot(ref_x, reference_data[joint], 'r--', linewidth=2,
                                   label='Reference')
                
                axes[i].set_ylabel(f'{joint.capitalize()} Angle (deg)')
                axes[i].legend()
                axes[i].grid(True)
        
        # Set common x-label
        axes[-1].set_xlabel(xlabel)
        
        # Set title
        fig.suptitle('Joint Angles', fontsize=16)
        fig.tight_layout()
        
        # Save figure if requested
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_ground_forces(self, results: SimulationResults,
                         normalize_gait_cycle: bool = True,
                         show_reference: bool = False,
                         reference_data: Optional[Dict[str, np.ndarray]] = None,
                         save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Plot ground reaction forces over time or gait cycle.
        
        Args:
            results: Simulation results
            normalize_gait_cycle: Whether to normalize time to gait cycle (0-100%)
            show_reference: Whether to show reference data
            reference_data: Reference ground force data
            save_path: Path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # Get time or gait cycle percentage
        if normalize_gait_cycle:
            x = np.linspace(0, 100, len(results.time))
            xlabel = 'Gait Cycle (%)'
        else:
            x = results.time
            xlabel = 'Time (s)'
        
        # Plot vertical force
        if 'vertical' in results.ground_forces:
            axes[0].plot(x, results.ground_forces['vertical'], 'b-', linewidth=2,
                       label='Simulation')
            
            # Plot reference data if available
            if show_reference and reference_data is not None:
                if 'vertical' in reference_data:
                    # Resample reference data to match simulation time points
                    if normalize_gait_cycle:
                        ref_x = np.linspace(0, 100, len(reference_data['vertical']))
                    else:
                        ref_x = np.linspace(results.time[0], results.time[-1],
                                          len(reference_data['vertical']))
                    
                    axes[0].plot(ref_x, reference_data['vertical'], 'r--', linewidth=2,
                               label='Reference')
            
            axes[0].set_ylabel('Vertical Force (N)')
            axes[0].legend()
            axes[0].grid(True)
        
        # Plot horizontal force
        if 'horizontal' in results.ground_forces:
            axes[1].plot(x, results.ground_forces['horizontal'], 'b-', linewidth=2,
                       label='Simulation')
            
            # Plot reference data if available
            if show_reference and reference_data is not None:
                if 'horizontal' in reference_data:
                    # Resample reference data to match simulation time points
                    if normalize_gait_cycle:
                        ref_x = np.linspace(0, 100, len(reference_data['horizontal']))
                    else:
                        ref_x = np.linspace(results.time[0], results.time[-1],
                                          len(reference_data['horizontal']))
                    
                    axes[1].plot(ref_x, reference_data['horizontal'], 'r--', linewidth=2,
                               label='Reference')
            
            axes[1].set_ylabel('Horizontal Force (N)')
            axes[1].legend()
            axes[1].grid(True)
        
        # Set common x-label
        axes[1].set_xlabel(xlabel)
        
        # Set title
        fig.suptitle('Ground Reaction Forces', fontsize=16)
        fig.tight_layout()
        
        # Save figure if requested
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_muscle_activations(self, results: SimulationResults,
                              muscles: Optional[List[str]] = None,
                              normalize_gait_cycle: bool = True,
                              save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Plot muscle activations over time or gait cycle.
        
        Args:
            results: Simulation results
            muscles: List of muscles to plot. If None, plots all muscles
            normalize_gait_cycle: Whether to normalize time to gait cycle (0-100%)
            save_path: Path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        # Get muscles to plot
        if muscles is None:
            muscles = list(results.muscle_activations.keys())
        
        # Create figure
        n_muscles = len(muscles)
        n_rows = (n_muscles + 1) // 2  # Ceiling division
        fig, axes = plt.subplots(n_rows, 2, figsize=self.figsize, sharex=True)
        axes = axes.flatten()
        
        # Get time or gait cycle percentage
        if normalize_gait_cycle:
            x = np.linspace(0, 100, len(results.time))
            xlabel = 'Gait Cycle (%)'
        else:
            x = results.time
            xlabel = 'Time (s)'
        
        # Plot each muscle
        for i, muscle in enumerate(muscles):
            if i < len(axes) and muscle in results.muscle_activations:
                axes[i].plot(x, results.muscle_activations[muscle], 'b-', linewidth=2)
                axes[i].set_ylabel(f'{muscle}')
                axes[i].set_ylim(0, 1)
                axes[i].grid(True)
        
        # Hide unused axes
        for i in range(n_muscles, len(axes)):
            axes[i].set_visible(False)
        
        # Set common x-label
        for i in range(min(n_muscles, len(axes))):
            if i // 2 == n_rows - 1:  # Last row
                axes[i].set_xlabel(xlabel)
        
        # Set title
        fig.suptitle('Muscle Activations', fontsize=16)
        fig.tight_layout()
        
        # Save figure if requested
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compare_simulations(self, results_list: List[SimulationResults],
                          labels: List[str],
                          plot_type: str = 'joint_angles',
                          normalize_gait_cycle: bool = True,
                          save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Compare multiple simulations.
        
        Args:
            results_list: List of simulation results
            labels: Labels for each simulation
            plot_type: Type of plot ('joint_angles', 'ground_forces', 'muscle_activations')
            normalize_gait_cycle: Whether to normalize time to gait cycle (0-100%)
            save_path: Path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        if len(results_list) != len(labels):
            raise ValueError("Number of results must match number of labels")
        
        if plot_type == 'joint_angles':
            # Get common joints
            joints = set()
            for results in results_list:
                joints.update(results.joint_angles.keys())
            joints = sorted(list(joints))
            
            # Create figure
            fig, axes = plt.subplots(len(joints), 1, figsize=self.figsize, sharex=True)
            if len(joints) == 1:
                axes = [axes]
            
            # Plot each joint for each simulation
            for i, joint in enumerate(joints):
                for j, (results, label) in enumerate(zip(results_list, labels)):
                    if joint in results.joint_angles:
                        # Get time or gait cycle percentage
                        if normalize_gait_cycle:
                            x = np.linspace(0, 100, len(results.time))
                        else:
                            x = results.time
                        
                        axes[i].plot(x, results.joint_angles[joint], linewidth=2,
                                   label=label)
                
                axes[i].set_ylabel(f'{joint.capitalize()} Angle (deg)')
                axes[i].legend()
                axes[i].grid(True)
            
            # Set common x-label
            if normalize_gait_cycle:
                axes[-1].set_xlabel('Gait Cycle (%)')
            else:
                axes[-1].set_xlabel('Time (s)')
            
            # Set title
            fig.suptitle('Joint Angles Comparison', fontsize=16)
        
        elif plot_type == 'ground_forces':
            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
            
            # Plot vertical force for each simulation
            for j, (results, label) in enumerate(zip(results_list, labels)):
                if 'vertical' in results.ground_forces:
                    # Get time or gait cycle percentage
                    if normalize_gait_cycle:
                        x = np.linspace(0, 100, len(results.time))
                    else:
                        x = results.time
                    
                    axes[0].plot(x, results.ground_forces['vertical'], linewidth=2,
                               label=label)
            
            axes[0].set_ylabel('Vertical Force (N)')
            axes[0].legend()
            axes[0].grid(True)
            
            # Plot horizontal force for each simulation
            for j, (results, label) in enumerate(zip(results_list, labels)):
                if 'horizontal' in results.ground_forces:
                    # Get time or gait cycle percentage
                    if normalize_gait_cycle:
                        x = np.linspace(0, 100, len(results.time))
                    else:
                        x = results.time
                    
                    axes[1].plot(x, results.ground_forces['horizontal'], linewidth=2,
                               label=label)
            
            axes[1].set_ylabel('Horizontal Force (N)')
            axes[1].legend()
            axes[1].grid(True)
            
            # Set common x-label
            if normalize_gait_cycle:
                axes[1].set_xlabel('Gait Cycle (%)')
            else:
                axes[1].set_xlabel('Time (s)')
            
            # Set title
            fig.suptitle('Ground Reaction Forces Comparison', fontsize=16)
        
        elif plot_type == 'muscle_activations':
            # Get common muscles
            muscles = set()
            for results in results_list:
                muscles.update(results.muscle_activations.keys())
            muscles = sorted(list(muscles))
            
            # Limit to top 6 muscles for readability
            if len(muscles) > 6:
                muscles = muscles[:6]
            
            # Create figure
            n_muscles = len(muscles)
            n_rows = (n_muscles + 1) // 2  # Ceiling division
            fig, axes = plt.subplots(n_rows, 2, figsize=self.figsize, sharex=True)
            axes = axes.flatten()
            
            # Plot each muscle for each simulation
            for i, muscle in enumerate(muscles):
                if i < len(axes):
                    for j, (results, label) in enumerate(zip(results_list, labels)):
                        if muscle in results.muscle_activations:
                            # Get time or gait cycle percentage
                            if normalize_gait_cycle:
                                x = np.linspace(0, 100, len(results.time))
                            else:
                                x = results.time
                            
                            axes[i].plot(x, results.muscle_activations[muscle], linewidth=2,
                                       label=label)
                    
                    axes[i].set_ylabel(f'{muscle}')
                    axes[i].set_ylim(0, 1)
                    axes[i].legend()
                    axes[i].grid(True)
            
            # Hide unused axes
            for i in range(n_muscles, len(axes)):
                axes[i].set_visible(False)
            
            # Set common x-label
            for i in range(min(n_muscles, len(axes))):
                if i // 2 == n_rows - 1:  # Last row
                    if normalize_gait_cycle:
                        axes[i].set_xlabel('Gait Cycle (%)')
                    else:
                        axes[i].set_xlabel('Time (s)')
            
            # Set title
            fig.suptitle('Muscle Activations Comparison', fontsize=16)
        
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        fig.tight_layout()
        
        # Save figure if requested
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig 