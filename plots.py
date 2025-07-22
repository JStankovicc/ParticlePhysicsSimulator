#!/usr/bin/env python3
"""
This module provides professional visualization tools for High Energy Physics (HEP)
analysis results, creating a suite of plots for detector performance, statistical
analysis, and overall simulation summaries.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import os

# Set professional style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HEPVisualizer:
    """
    Generates a variety of professional plots for HEP analysis results.
    """
    
    def __init__(self, output_dir: str = "output_graphs"):
        """Initializes the HEPVisualizer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # A professional color scheme for different particle types and analysis elements.
        self.colors = {
            'electron': '#00FFFF',      # Cyan
            'proton': '#FF00FF',        # Magenta
            'neutron': '#32CD32',       # Lime
            'photon': '#FFD700',        # Gold
            'quark_up': '#FF69B4',      # Hot Pink
            'quark_down': '#FF7F50',    # Coral
            'muon': '#9370DB',          # Medium Purple
            'background': '#2C3E50',    # Dark Blue-Gray
            'signal': '#E74C3C',        # Red
            'fit': '#F39C12'            # Orange
        }
        
        # Default matplotlib settings for creating professional-looking plots.
        self.fig_params = {
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': 300
        }
        plt.rcParams.update(self.fig_params)
    
    def plot_detector_layout(self, detector_data: Dict, save_name: str = "detector_layout.png"):
        """Plots the detector layout, including hits and reconstructed tracks.
        
        Args:
            detector_data: Detector configuration and hits
            save_name: Output filename
        """
        hits = detector_data.get('hits', [])
        tracks = detector_data.get('tracks', [])
        
        # If the number of hits is excessively large, fall back to a text summary
        # to avoid memory errors during plotting.
        if len(hits) > 10000:  # Safety threshold (approximate for very large datasets)
            print(f"Warning: Too many hits ({len(hits)}) for safe visualization. Creating summary instead.")
            return self._create_detector_summary(detector_data, save_name)
        
        # Use a smaller figure size to reduce memory consumption.
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Draw the concentric circles representing the detector layers.
        layers = detector_data.get('layers', {})
        for layer_name, layer_config in layers.items():
            r_min, r_max = layer_config['radius_range']
            
            # Skip huge outer layers that won't be visible in current zoom (±40 μm)
            if r_max > 40:
                continue
            
            # Draw detector layer as annulus
            circle_outer = Circle((0, 0), r_max, fill=False, 
                                edgecolor='black', linewidth=2, alpha=0.7)
            circle_inner = Circle((0, 0), r_min, fill=False, 
                                edgecolor='black', linewidth=1, alpha=0.5)
            ax.add_patch(circle_outer)
            ax.add_patch(circle_inner)
            
            # Label each detector layer.
            label_radius = (r_min + r_max) / 2
            ax.text(label_radius * 0.7, label_radius * 0.7, 
                   layer_name.replace('_', ' ').title(),
                   fontsize=9, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # To manage memory, plot only a random sample of the detector hits.
        max_hits_to_plot = 100
        
        if len(hits) > max_hits_to_plot:
            print(f"Warning: Too many hits ({len(hits)}), displaying random sample of {max_hits_to_plot}")
            # Take a uniform random sample of hits.
            import random
            hits_sample = random.sample(hits, max_hits_to_plot)
        else:
            hits_sample = hits
        
        # Group hits by particle type for scatter plot
        hit_groups = {}
        for hit in hits_sample:
            particle_type = hit.particle_type
            if particle_type not in hit_groups:
                hit_groups[particle_type] = {'x': [], 'y': []}
            hit_groups[particle_type]['x'].append(hit.position[0])
            hit_groups[particle_type]['y'].append(hit.position[1])
        
        # Use a scatter plot for hits, which is much more memory-efficient than
        # creating individual Circle patches for each hit.
        for particle_type, coords in hit_groups.items():
            color = self.colors.get(particle_type, 'gray')
            ax.scatter(coords['x'], coords['y'], c=color, s=15, alpha=0.6, 
                      edgecolors='none')
        
        # Limit the number of plotted tracks to avoid visual clutter.
        max_tracks_to_plot = 15
        
        if len(tracks) > max_tracks_to_plot:
            print(f"Warning: Too many tracks ({len(tracks)}), displaying first {max_tracks_to_plot}")
            tracks_sample = tracks[:max_tracks_to_plot]
        else:
            tracks_sample = tracks
            
        for track in tracks_sample:
            track_hits = track.get('hits', [])
            if len(track_hits) >= 2:
                x_coords = [hit.position[0] for hit in track_hits]
                y_coords = [hit.position[1] for hit in track_hits]
                particle_type = track.get('particle_type', 'unknown')
                color = self.colors.get(particle_type, 'gray')
                
                # Draw track line
                ax.plot(x_coords, y_coords, color=color, linewidth=1, 
                       alpha=0.7, linestyle='-')
        
        # Set plot formatting and labels.
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_aspect('equal')
        ax.set_xlabel('X Position (μm)', fontsize=12)
        ax.set_ylabel('Y Position (μm)', fontsize=12)
        ax.set_title('HEP Detector Layout', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Create a simplified legend with only the main particle types.
        main_particle_types = ['electron', 'proton', 'neutron', 'photon']
        legend_elements = []
        for particle_type in main_particle_types:
            if particle_type in self.colors:
                color = self.colors[particle_type]
                legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=6, 
                                                label=particle_type.title()))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add a text box indicating that the hits and tracks have been sampled.
        info_text = f"Showing {len(hits_sample)} of {len(hits)} hits, {len(tracks_sample)} of {len(tracks)} tracks"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
        plt.tight_layout()
        # Use a low DPI to further reduce memory usage.
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=75, bbox_inches='tight')
        plt.close()
    
    def _create_detector_summary(self, detector_data: Dict, save_name: str):
        """Creates a plain text summary of detector data, used as a fallback for large datasets."""
        
        # Get data
        hits = detector_data.get('hits', [])
        tracks = detector_data.get('tracks', [])
        layers = detector_data.get('layers', {})
        
        # Count hits by particle type
        hit_counts = {}
        for hit in hits:
            particle_type = hit.particle_type
            hit_counts[particle_type] = hit_counts.get(particle_type, 0) + 1
        
        # Count tracks by particle type
        track_counts = {}
        for track in tracks:
            particle_type = track.get('particle_type', 'unknown')
            track_counts[particle_type] = track_counts.get(particle_type, 0) + 1
        
        # Create summary text
        summary_text = f"""
HEP DETECTOR ANALYSIS SUMMARY
{'='*50}

DETECTOR CONFIGURATION:
• Detector layers: {len(layers)}
• Layer names: {', '.join(layers.keys())}

DETECTION STATISTICS:
• Total hits detected: {len(hits):,}
• Total tracks reconstructed: {len(tracks)}

HITS BY PARTICLE TYPE:
"""
        
        for particle_type, count in sorted(hit_counts.items()):
            percentage = (count / len(hits) * 100) if hits else 0
            summary_text += f"• {particle_type}: {count:,} hits ({percentage:.1f}%)\n"
        
        summary_text += f"""
TRACKS BY PARTICLE TYPE:
"""
        
        for particle_type, count in sorted(track_counts.items()):
            percentage = (count / len(tracks) * 100) if tracks else 0
            summary_text += f"• {particle_type}: {count} tracks ({percentage:.1f}%)\n"
        
        summary_text += f"""
DETECTOR EFFICIENCY:
• Detection rate: {len(tracks)/max(1, len(hits))*100:.1f}%
• Average hits per track: {len(hits)/max(1, len(tracks)):.1f}

NOTE: Graphical visualization skipped due to memory constraints.
This summary provides key detector analysis results.
"""
        
        # Save the summary to a text file.
        base_name = os.path.splitext(save_name)[0]
        txt_name = f"{base_name}.txt"
        txt_path = os.path.join(self.output_dir, txt_name)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"Detector summary saved to {txt_path}")
        
        # No PNG is generated in this case.
        return
    
    def plot_invariant_mass_spectrum(self, analysis_results: Dict, 
                                   save_name: str = "invariant_mass_spectrum.png"):
        """Plots the invariant mass spectrum, highlighting any detected resonance peaks.
        
        Args:
            analysis_results: HEP analysis results
            save_name: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        invariant_masses = analysis_results.get('invariant_masses', [])
        resonances = analysis_results.get('resonances', [])
        
        if not invariant_masses:
            ax1.text(0.5, 0.5, 'No invariant mass data available', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=14)
            ax1.set_title('Invariant Mass Spectrum', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # Plot the main histogram of the invariant mass spectrum.
        bins = 50
        counts, bin_edges, patches = ax1.hist(invariant_masses, bins=bins, 
                                            alpha=0.7, color='skyblue', 
                                            edgecolor='black', linewidth=0.5)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Highlight significant resonance peaks on the histogram.
        for resonance in resonances:
            mass = resonance['mass']
            significance = resonance['significance']
            
            # Find corresponding bin
            bin_idx = np.argmin(np.abs(bin_centers - mass))
            
            # Color the peak's bar red to make it stand out.
            if bin_idx < len(patches):
                patches[bin_idx].set_facecolor('red')
                patches[bin_idx].set_alpha(0.8)
            
            # Add an annotation indicating the statistical significance of the peak.
            ax1.annotate(f'{significance:.1f}σ', 
                        xy=(mass, counts[bin_idx]), 
                        xytext=(mass, counts[bin_idx] + max(counts) * 0.1),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=12, ha='center', color='red', fontweight='bold')
        
        ax1.set_xlabel('Invariant Mass (Energy Units)', fontsize=14)
        ax1.set_ylabel('Counts', fontsize=14)
        ax1.set_title('Invariant Mass Spectrum with Resonance Peaks', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Create a second subplot to provide a zoomed-in view of the main resonance.
        if resonances:
            # Focus on the most significant resonance
            main_resonance = max(resonances, key=lambda r: r['significance'])
            mass_center = main_resonance['mass']
            mass_range = np.ptp(invariant_masses) * 0.1  # 10% of total range
            
            # Filter the data to the region around the main resonance.
            zoom_mask = (np.array(invariant_masses) >= mass_center - mass_range) & \
                       (np.array(invariant_masses) <= mass_center + mass_range)
            zoom_masses = np.array(invariant_masses)[zoom_mask]
            
            if len(zoom_masses) > 0:
                ax2.hist(zoom_masses, bins=20, alpha=0.7, color='lightcoral', 
                        edgecolor='black', linewidth=0.5)
                ax2.axvline(mass_center, color='red', linestyle='--', linewidth=2, 
                          label=f'Peak at {mass_center:.2f}')
                ax2.set_xlabel('Invariant Mass (Energy Units)', fontsize=14)
                ax2.set_ylabel('Counts', fontsize=14)
                ax2.set_title('Zoomed View of Main Resonance', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
        
        plt.tight_layout()
        # Use lower DPI to reduce memory usage
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=100, bbox_inches='tight')
        plt.close()
    
    def plot_detector_efficiency(self, efficiency_data: Dict, 
                               save_name: str = "detector_efficiency.png"):
        """Plots the detector's efficiency, both overall and broken down by particle type.
        
        Args:
            efficiency_data: Efficiency analysis results
            save_name: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot the overall detector efficiency in the first subplot.
        overall_eff = efficiency_data.get('overall_efficiency', 0)
        overall_err = efficiency_data.get('overall_error', 0)
        
        ax1.bar(['Overall Efficiency'], [overall_eff], 
               yerr=[overall_err], capsize=10, 
               color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Efficiency', fontsize=14)
        ax1.set_title('Overall Detector Efficiency', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Add a text annotation with the efficiency value.
        ax1.text(0, overall_eff + overall_err + 0.05, 
                f'{overall_eff:.3f} ± {overall_err:.3f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Plot the efficiency for each particle type in the second subplot.
        eff_by_type = efficiency_data.get('efficiency_by_type', {})
        if eff_by_type:
            particle_types = list(eff_by_type.keys())
            efficiencies = [eff_by_type[ptype]['efficiency'] for ptype in particle_types]
            errors = [eff_by_type[ptype]['error'] for ptype in particle_types]
            
            colors = [self.colors.get(ptype, 'gray') for ptype in particle_types]
            
            bars = ax2.bar(particle_types, efficiencies, yerr=errors, 
                          capsize=5, color=colors, alpha=0.7, edgecolor='black')
            
            # Add text annotations for each bar.
            for i, (bar, eff, err) in enumerate(zip(bars, efficiencies, errors)):
                ax2.text(bar.get_x() + bar.get_width()/2, eff + err + 0.02,
                        f'{eff:.2f}', ha='center', va='bottom', fontsize=10)
            
            ax2.set_ylabel('Efficiency', fontsize=14)
            ax2.set_xlabel('Particle Type', fontsize=14)
            ax2.set_title('Efficiency by Particle Type', fontsize=14, fontweight='bold')
            ax2.set_ylim(0, 1.1)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        # Use lower DPI to reduce memory usage
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=100, bbox_inches='tight')
        plt.close()
    
    def plot_statistical_analysis(self, stats_data: Dict, 
                                save_name: str = "statistical_analysis.png"):
        """Creates a plot summarizing various statistical analysis results.
        
        Args:
            stats_data: Statistical analysis results
            save_name: Output filename
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Subplot 1: Significance of detected resonances.
        resonances = stats_data.get('resonances', [])
        if resonances:
            masses = [r['mass'] for r in resonances]
            significances = [r['significance'] for r in resonances]
            
            # Color-code the points based on their significance level (e.g., hint, evidence, discovery).
            colors = []
            for sig in significances:
                if sig >= 5.0:
                    colors.append('red')      # Discovery
                elif sig >= 3.0:
                    colors.append('orange')   # Evidence
                elif sig >= 2.0:
                    colors.append('yellow')   # Hint
                else:
                    colors.append('gray')     # Background
            
            ax1.scatter(masses, significances, c=colors, s=100, alpha=0.7, edgecolors='black')
            ax1.axhline(y=5.0, color='red', linestyle='--', label='Discovery (5σ)')
            ax1.axhline(y=3.0, color='orange', linestyle='--', label='Evidence (3σ)')
            ax1.axhline(y=2.0, color='yellow', linestyle='--', label='Hint (2σ)')
            
            ax1.set_xlabel('Invariant Mass', fontsize=14)
            ax1.set_ylabel('Statistical Significance (σ)', fontsize=14)
            ax1.set_title('Statistical Significance of Resonances', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
            ax1.grid(True, alpha=0.3)
        
        # Subplot 2: A synthesized energy spectrum distribution.
        energy_stats = stats_data.get('energy_spectrum', {})
        if energy_stats and energy_stats.get('count', 0) > 0:
            # Create mock energy distribution for visualization
            mean_energy = energy_stats['mean']
            std_energy = energy_stats['std']
            
            x = np.linspace(0, mean_energy + 3*std_energy, 100)
            y = np.exp(-(x - mean_energy)**2 / (2 * std_energy**2))
            
            ax2.plot(x, y, 'b-', linewidth=2, label='Energy Distribution')
            ax2.axvline(mean_energy, color='red', linestyle='--', 
                       label=f'Mean: {mean_energy:.2f}')
            ax2.fill_between(x, y, alpha=0.3)
            
            ax2.set_xlabel('Energy', fontsize=14)
            ax2.set_ylabel('Normalized Counts', fontsize=14)
            ax2.set_title('Energy Spectrum Distribution', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Distribution of chi-squared values from goodness-of-fit tests.
        chi_squared_data = stats_data.get('chi_squared_tests', [])
        if chi_squared_data:
            chi_values = [test['chi_squared'] for test in chi_squared_data]
            p_values = [test['p_value'] for test in chi_squared_data]
            
            ax3.hist(chi_values, bins=20, alpha=0.7, color='lightgreen', 
                    edgecolor='black', linewidth=0.5)
            ax3.set_xlabel('Chi-squared Value', fontsize=14)
            ax3.set_ylabel('Frequency', fontsize=14)
            ax3.set_title('Chi-squared Test Results', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Distribution of p-values from the chi-squared tests.
        if chi_squared_data:
            ax4.hist(p_values, bins=20, alpha=0.7, color='lightcoral', 
                    edgecolor='black', linewidth=0.5)
            ax4.axvline(0.05, color='red', linestyle='--', label='α = 0.05')
            ax4.set_xlabel('P-value', fontsize=14)
            ax4.set_ylabel('Frequency', fontsize=14)
            ax4.set_title('P-value Distribution', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Use lower DPI to reduce memory usage for large figure
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=100, bbox_inches='tight')
        plt.close()
    
    def create_hep_summary_report(self, simulation_data: Dict, detector_data: Dict, 
                                analysis_results: Dict, save_name: str = "hep_analysis_report.png"):
        """Creates a comprehensive summary report with multiple subplots for HEP analysis.
        
        Args:
            simulation_data: Simulation results
            detector_data: Detector data
            analysis_results: Analysis results
            save_name: Output filename
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Use a GridSpec for a compact and well-spaced layout.
        gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.3)  # 4 rows, better spacing
        
        # --- Main Title ---
        fig.suptitle('High Energy Physics Analysis Report', fontsize=24, fontweight='bold', y=0.95)
        
        # --- Row 0: Detector Overview ---
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_mini_detector(ax1, detector_data)
        
        # Invariant mass spectrum (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_mini_mass_spectrum(ax2, analysis_results)
        
        # Efficiency summary (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_mini_efficiency(ax3, detector_data)
        
        # --- Row 1: Key Analysis Metrics ---
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_mini_energy_dist(ax4, analysis_results)
        
        # Significance plot (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_mini_significance(ax5, analysis_results)
         
        # Statistics table (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_statistics_table(ax6, simulation_data, analysis_results)
         
        # --- Row 2: Statistical Tests and Trajectories ---
        ax8 = fig.add_subplot(gs[2, 0])
        self._plot_mini_chi_squared(ax8, analysis_results)
 
        ax9 = fig.add_subplot(gs[2, 1])
        self._plot_mini_p_values(ax9, analysis_results)
 
        # Particle trajectories and detector response (row2, col2)
        ax10 = fig.add_subplot(gs[2, 2])
        self._plot_particle_tracks_summary(ax10, simulation_data, detector_data)
 
        # --- Row 3: Interaction Timeline ---
        ax11 = fig.add_subplot(gs[3, :])
        self._plot_mini_interaction_timeline(ax11, simulation_data)
        
        plt.savefig(os.path.join(self.output_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_mini_detector(self, ax, detector_data):
        """Helper method to plot a miniature, auto-scaled detector layout."""
        # Simplified detector visualization (auto-scaled)
        layers = detector_data.get('layers', {})
        max_radius = 0.0
        for layer_name, layer_config in layers.items():
            r_min, r_max = layer_config['radius_range']
            max_radius = max(max_radius, r_max)
            circle = Circle((0, 0), r_max, fill=False, edgecolor='black', linewidth=1)
            ax.add_patch(circle)
        
        if max_radius == 0:
            max_radius = 1.0
        margin = 0.1 * max_radius
        ax.set_xlim(-(max_radius + margin), max_radius + margin)
        ax.set_ylim(-(max_radius + margin), max_radius + margin)
        ax.set_aspect('equal')
        ax.set_title('Detector Layout', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_mini_mass_spectrum(self, ax, analysis_results):
        """Helper method to plot a miniature invariant mass spectrum."""
        invariant_masses = analysis_results.get('invariant_masses', [])
        if invariant_masses:
            ax.hist(invariant_masses, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Invariant Mass')
            ax.set_ylabel('Counts')
        ax.set_title('Invariant Mass Spectrum', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_mini_efficiency(self, ax, detector_data):
        """Helper method to plot a miniature detector efficiency chart."""
        summary = detector_data.get('summary', {})
        efficiency = summary.get('detector_efficiency', 0)
        
        ax.bar(['Efficiency'], [efficiency], color='steelblue', alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_title('Detector Efficiency', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_mini_energy_dist(self, ax, analysis_results):
        """Helper method to plot a miniature energy distribution."""
        energy_stats = analysis_results.get('energy_spectrum', {})
        if energy_stats and energy_stats.get('count', 0) > 0:
            mean_energy = energy_stats['mean']
            std_energy = energy_stats['std']
            
            x = np.linspace(0, mean_energy + 2*std_energy, 50)
            y = np.exp(-(x - mean_energy)**2 / (2 * std_energy**2))
            ax.plot(x, y, 'b-', linewidth=2)
            ax.fill_between(x, y, alpha=0.3)
        
        ax.set_title('Energy Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_mini_significance(self, ax, analysis_results):
        """Helper method to plot a miniature statistical significance chart."""
        resonances = analysis_results.get('resonances', [])
        if resonances:
            significances = [r['significance'] for r in resonances]
            ax.bar(range(len(significances)), significances, color='red', alpha=0.7)
            ax.axhline(y=5.0, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=3.0, color='orange', linestyle='--', alpha=0.5)
        
        ax.set_title('Statistical Significance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Significance (σ)')
        ax.grid(True, alpha=0.3)
    
    def _plot_statistics_table(self, ax, simulation_data, analysis_results):
        """Helper method to display key simulation statistics in a table."""
        ax.axis('off')
        
        # Prepare statistics
        stats = [
            ['Total Particles', simulation_data.get('total_particles', 0)],
            ['Active Particles', simulation_data.get('active_particles', 0)],
            ['Total Collisions', simulation_data.get('total_collisions', 0)],
            ['Resonances Found', len(analysis_results.get('resonances', []))],
            ['Analysis Quality', analysis_results.get('analysis_quality', 'N/A')]
        ]
        
        # Create table
        table = ax.table(cellText=stats, colLabels=['Parameter', 'Value'],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax.set_title('Simulation Statistics', fontsize=12, fontweight='bold')
    
    def _plot_particle_tracks_summary(self, ax, simulation_data, detector_data):
        """Helper method to plot a summary of particle trajectories."""
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_aspect('equal')
        ax.set_xlabel('X Position (μm)', fontsize=14)
        ax.set_ylabel('Y Position (μm)', fontsize=14)
        ax.set_title('Particle Trajectories', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot actual particle trajectories from simulation data
        history_data = simulation_data.get('history', {})
        time_data = history_data.get('time', [])
        
        # Get particle trajectories from simulation if available
        trajectories_plotted = 0
        max_trajectories = 15  # Limit to prevent overcrowding
        
        # Try to get particles from the simulation data
        if 'particles' in simulation_data:
            particles = simulation_data['particles']
        elif 'particles' in history_data:
            particles = history_data['particles']
        else:
            particles = []
        
        # If direct particle trajectory data is not available, fall back to plotting
        # the reconstructed detector tracks.
        if not particles:
            tracks = detector_data.get('tracks', [])
            for i, track in enumerate(tracks[:max_trajectories]):
                hits = track.get('hits', [])
                if len(hits) >= 2:
                    x_coords = [hit.position[0] for hit in hits]
                    y_coords = [hit.position[1] for hit in hits]
                    particle_type = track.get('particle_type', 'unknown')
                    color = self.colors.get(particle_type, 'gray')
                    
                    ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
                    trajectories_plotted += 1
        else:
            # Plot the actual trajectories from the simulation history.
            for i, particle in enumerate(particles[:max_trajectories]):
                if hasattr(particle, 'history') and particle.history.get('position'):
                    positions = particle.history['position']
                    if len(positions) >= 2:
                        x_coords = [pos[0] for pos in positions]
                        y_coords = [pos[1] for pos in positions]
                        
                        # Get particle color
                        if hasattr(particle, 'particle_type'):
                            color = self.colors.get(particle.particle_type, 'gray')
                        else:
                            color = getattr(particle, 'color', 'gray')
                        
                        ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
                        trajectories_plotted += 1
        
        # Add outlines of the detector layers for spatial context.
        layers = detector_data.get('layers', {})
        for layer_name, layer_config in layers.items():
            r_min, r_max = layer_config['radius_range']
            # Only draw layers that are within the plot's current view.
            if r_max <= 30:
                circle = Circle((0, 0), r_max, fill=False, edgecolor='black', 
                              linewidth=1, alpha=0.3, linestyle='--')
                ax.add_patch(circle)
        
        # Dynamically adjust the axis limits to fit the plotted data.
        all_x = []
        all_y = []
        for line in ax.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            all_x.extend(xdata)
            all_y.extend(ydata)
        if all_x and all_y:
            xmin, xmax = min(all_x), max(all_x)
            ymin, ymax = min(all_y), max(all_y)
            max_range = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
            margin = 0.1 * max_range if max_range != 0 else 1.0
            ax.set_xlim(-(max_range + margin), max_range + margin)
            ax.set_ylim(-(max_range + margin), max_range + margin)
        
        # Add info text
        if trajectories_plotted > 0:
            ax.text(0.02, 0.98, f'Trajectories shown: {trajectories_plotted}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No trajectory data\navailable', ha='center', va='center', 
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))

    def _plot_mini_chi_squared(self, ax, analysis_results):
        """Helper method to plot a miniature chi-squared distribution."""
        chi_squared_data = analysis_results.get('chi_squared_tests', [])
        if not chi_squared_data:
            ax.text(0.5, 0.5, 'No chi-squared data\navailable', ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
            ax.set_title('Chi-squared Distribution', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return
        
        # Extract and validate chi-squared values from the analysis results.
        chi_values = []
        for test in chi_squared_data:
            if isinstance(test, dict) and 'chi_squared' in test:
                chi_val = test['chi_squared']
                if isinstance(chi_val, (int, float)) and not np.isnan(chi_val) and chi_val >= 0:
                    chi_values.append(chi_val)
        
        if not chi_values:
            ax.text(0.5, 0.5, 'No valid chi-squared\ndata available', ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            ax.hist(chi_values, bins=min(10, len(chi_values)), alpha=0.7, color='lightgreen', edgecolor='black')
            ax.set_xlabel('Chi-squared Value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            
        ax.set_title('Chi-squared Distribution', fontsize=12)
        ax.grid(True, alpha=0.3)

    def _plot_mini_p_values(self, ax, analysis_results):
        """Helper method to plot a miniature p-value distribution."""
        chi_squared_data = analysis_results.get('chi_squared_tests', [])
        if not chi_squared_data:
            ax.text(0.5, 0.5, 'No p-value data\navailable', ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
            ax.set_title('P-value Distribution', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return
        
        # Extract and validate p-values from the analysis results.
        p_values = []
        for test in chi_squared_data:
            if isinstance(test, dict) and 'p_value' in test:
                p_val = test['p_value']
                if isinstance(p_val, (int, float)) and not np.isnan(p_val) and 0 <= p_val <= 1:
                    p_values.append(p_val)
        
        if not p_values:
            ax.text(0.5, 0.5, 'No valid p-value\ndata available', ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            ax.hist(p_values, bins=min(10, len(p_values)), alpha=0.7, color='lightcoral', edgecolor='black')
            ax.axvline(0.05, color='red', linestyle='--', label='α = 0.05')
            ax.set_xlabel('P-value', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend(fontsize=8)
            
        ax.set_title('P-value Distribution', fontsize=12)
        ax.grid(True, alpha=0.3)

    def _plot_key_metrics_text(self, ax, simulation_data, analysis_results):
        """Helper method to display key metrics as text."""
        ax.axis('off')
        
        # Safely extract metrics, providing default values if data is missing or invalid.
        resonances = analysis_results.get('resonances', [])
        significance_values = []
        for r in resonances:
            if isinstance(r, dict) and 'significance' in r:
                sig = r['significance']
                if isinstance(sig, (int, float)) and not np.isnan(sig):
                    significance_values.append(sig)
        
        max_sig = max(significance_values) if significance_values else 0
        
        # Safely extract other metrics
        total_events = simulation_data.get('total_events', 0)
        if not isinstance(total_events, (int, float)):
            total_events = 0
            
        overall_efficiency = analysis_results.get('overall_efficiency', 0)
        if not isinstance(overall_efficiency, (int, float)) or np.isnan(overall_efficiency):
            overall_efficiency = 0
            
        analysis_quality = analysis_results.get('analysis_quality', 'N/A')
        if not isinstance(analysis_quality, str):
            analysis_quality = 'N/A'
        
        metrics_text = (
            f"Total Events: {int(total_events)}\n"
            f"Avg Efficiency: {overall_efficiency:.2f}\n"
            f"Max Significance: {max_sig:.1f}σ\n"
            f"Analysis Quality: {analysis_quality}"
        )
        ax.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))

    def _plot_mini_interaction_timeline(self, ax, simulation_data):
        """Helper method to plot a miniature interaction timeline."""
        # Get data from simulation history
        history_data = simulation_data.get('history', {})
        time_data = history_data.get('time', [])
        active_data = history_data.get('active_particles', [])
        
        # Ensure that the data is in the expected list format.
        if not isinstance(time_data, list):
            time_data = []
        if not isinstance(active_data, list):
            active_data = []
        
        # If we have valid data, plot it
        if time_data and active_data:
            # Ensure data lengths match
            min_len = min(len(time_data), len(active_data))
            if min_len > 1:
                # Truncate to matching length
                time_data = time_data[:min_len]
                active_data = active_data[:min_len]
                
                # Plot the timeline
                ax.plot(time_data, active_data, 'b-', linewidth=2, alpha=0.8)
                ax.fill_between(time_data, active_data, alpha=0.3, color='blue')
                ax.set_xlabel('Time (s)', fontsize=10)
                ax.set_ylabel('Active Particles', fontsize=10)
                ax.set_title('Particle Interaction Timeline', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                total_particles = simulation_data.get('total_particles', len(active_data))
                final_active = active_data[-1] if active_data else 0
                ax.text(0.02, 0.98, f'Total: {total_particles}\nFinal: {final_active}', 
                       transform=ax.transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
            return
        
        # If active particle data is unavailable, fall back to plotting a timeline of events.
        collision_times = []
        decay_times = []
        
        if 'collisions' in history_data:
            collision_times = [event.get('time', 0) for event in history_data['collisions']]
        if 'decays' in history_data:
            decay_times = [event.get('time', 0) for event in history_data['decays']]
        
        if collision_times or decay_times:
            # Plot event timeline
            all_times = sorted(collision_times + decay_times)
            if all_times:
                max_time = max(all_times) if all_times else 1
                
                # Plot collision events
                if collision_times:
                    ax.scatter(collision_times, [1] * len(collision_times), 
                             color='red', s=30, alpha=0.8, label='Collisions', marker='o')
                
                # Plot decay events  
                if decay_times:
                    ax.scatter(decay_times, [0.5] * len(decay_times),
                             color='orange', s=30, alpha=0.8, label='Decays', marker='s')
                
                ax.set_xlabel('Time (s)', fontsize=10)
                ax.set_ylabel('Event Type', fontsize=10)
                ax.set_title('Interaction Events Timeline', fontsize=12, fontweight='bold')
                ax.set_xlim(0, max_time)
                ax.set_ylim(0, 1.5)
                ax.grid(True, alpha=0.3)
                
                if collision_times and decay_times:
                    ax.legend(fontsize=8)
                
                # Add event count
                total_events = len(collision_times) + len(decay_times)
                ax.text(0.02, 0.98, f'Events: {total_events}', 
                       transform=ax.transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
            return
        
        # If no data is available to plot, display a message on the subplot.
        ax.text(0.5, 0.5, 'No timeline data\navailable', ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
        ax.set_title('Interaction Timeline', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1) 