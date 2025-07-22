import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from typing import Dict, List, Optional
import os
from datetime import datetime

class Visualizer:
    """
    Handles visualization and animation of particle simulation results.
    """
    
    def __init__(self, simulation):
        """
        Initialize visualizer with simulation data.
        
        Args:
            simulation: Simulation object to visualize
        """
        self.simulation = simulation
        self.fig = None
        self.ax = None
        self.animation = None
        
    def animate(self, save_path: Optional[str] = None, interval: int = 50) -> None:
        """
        Create and display animation of particle motion.
        
        Args:
            save_path: Path to save animation (optional)
            interval: Animation interval in milliseconds
        """
        # Get particle trajectory data
        trajectories = self.simulation.get_particle_trajectories()
        
        if not trajectories:
            print("No particle trajectories to animate!")
            return
            
        # Set up matplotlib figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlabel('X Position (μm)')
        self.ax.set_ylabel('Y Position (μm)')
        self.title_text = self.ax.set_title('Particle Motion in Magnetic Field')
        self.ax.grid(True, alpha=0.3)
        
        # Collect all positions to determine plot scale
        all_positions_arr = []
        for traj in trajectories.values():
            if len(traj['positions']) > 0:
                all_positions_arr.extend(traj['positions'])
        all_positions_arr = np.array(all_positions_arr)
        
        # Use 95th percentile to set scale, ignoring outliers
        if len(all_positions_arr) > 0:
            x_95 = float(np.percentile(np.abs(all_positions_arr[:, 0]), 95))
            y_95 = float(np.percentile(np.abs(all_positions_arr[:, 1]), 95))
            max_position = max([x_95, y_95])
        else:
            max_position = 0.0
        print(f"Debug: 95th percentile max position = {max_position}")
        
        # Adjust particle and trail sizes based on spatial scale
        if max_position < 10.0:  # Small scale
            particle_size = 0.00001  # Very small for realistic scale
            trail_width = 0.05
            print(f"Debug: Using small scale - particle_size={particle_size}")
        else:  # Large scale
            particle_size = 300
            trail_width = 3.0
            print(f"Debug: Using large scale - particle_size={particle_size}")
        
        # Set axis limits using 99th percentile for margin
        if len(all_positions_arr) > 0:
            x_lim = float(np.percentile(np.abs(all_positions_arr[:, 0]), 99))
            y_lim = float(np.percentile(np.abs(all_positions_arr[:, 1]), 99))
            if max_position < 10.0:
                lim = max([10.0, x_lim, y_lim])
            else:
                lim = max([1000.0, x_lim, y_lim])
            self.ax.set_xlim(-lim, lim)
            self.ax.set_ylim(-lim, lim)
        else:
            lim = 10.0
            self.ax.set_xlim(-lim, lim)
            self.ax.set_ylim(-lim, lim)

        # Calculate particle radius based on axis span for visibility
        particle_radius = max(2.0, 0.02 * lim)  # 2% of span, min 2 μm
        
        # Initialize graphical objects for each particle (circle and trail)
        particle_circles = {}
        trail_lines = {}
        
        for particle_id, traj in trajectories.items():
            if len(traj['positions']) > 0:
                # Adjust circle radius for different scales
                if max_position < 10.0:
                    circle = Circle((0, 0), particle_radius * 0.2, facecolor=traj['color'], alpha=1.0, edgecolor='white', linewidth=1.0)
                else:
                    circle = Circle((0, 0), particle_radius, facecolor=traj['color'], alpha=1.0, edgecolor='white', linewidth=1.5)
                self.ax.add_patch(circle)
                particle_circles[particle_id] = circle
                
                # Create line object for particle trail
                line, = self.ax.plot([], [], color=traj['color'], alpha=0.9, linewidth=trail_width)
                trail_lines[particle_id] = line
                
        # Animation function
        def animate(frame):
            time_idx = min(frame, len(self.simulation.history['time']) - 1)
            current_time = self.simulation.history['time'][time_idx]
            
            # Update position and trail for each particle
            for particle_id, traj in trajectories.items():
                if particle_id in particle_circles:
                    if time_idx < len(traj['positions']):
                        position = traj['positions'][time_idx]
                        if np.all(np.isfinite(position)):
                            particle_circles[particle_id].center = position
                            trail_start = max(0, time_idx - 15)
                            trail_positions = traj['positions'][trail_start:time_idx + 1]
                            if len(trail_positions) > 0:
                                valid_trail = []
                                for pos in trail_positions:
                                    if np.all(np.isfinite(pos)):
                                        valid_trail.append(pos)
                                if len(valid_trail) > 1:
                                    valid_trail = np.array(valid_trail)
                                    trail_lines[particle_id].set_data(valid_trail[:, 0], valid_trail[:, 1])
                        # Keep active particles visible
                        particle_circles[particle_id].set_alpha(1.0)
                        trail_lines[particle_id].set_alpha(0.9)
                    else:
                        # Hide decayed particles
                        particle_circles[particle_id].set_alpha(0.0)
                        trail_lines[particle_id].set_alpha(0.0)
                        
            # Update title with current time and active particle count
            active_count = sum(1 for traj in trajectories.values() if time_idx < len(traj['positions']))
            new_title = f'Particle Motion (t = {current_time:.2f}s, active: {active_count}/{len(trajectories)})'
            self.title_text.set_text(new_title)
            
            # Return updated objects
            all_objects = list(particle_circles.values()) + list(trail_lines.values())
            return all_objects
            
        # Create animation using FuncAnimation
        frames = len(self.simulation.history['time'])
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=frames, interval=interval, 
            blit=False, repeat=True
        )
        
        # Save animation if path provided
        if save_path:
            self.save_animation(save_path)
            
        plt.show()
        
    def save_animation(self, path: str) -> None:
        """
        Save animation to file.
        
        Args:
            path: File path to save animation
        """
        if self.animation:
            print(f"Saving animation to {path}...")
            self.animation.save(path, writer='pillow', fps=20)
            print("Animation saved successfully!")
            
    def generate_plots(self, output_dir: str = "output_graphs") -> None:
        """Generate and save standard analysis plots.
        
        Args:
            output_dir: Directory to save plots
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get data from simulation
        trajectories = self.simulation.get_particle_trajectories()
        distances = self.simulation.get_interparticle_distances()
        
        # Generate all plots
        self._plot_positions_vs_time(trajectories, output_dir)
        self._plot_velocities_vs_time(trajectories, output_dir)
        self._plot_accelerations_vs_time(trajectories, output_dir)
        self._plot_system_energy(output_dir)
        self._plot_interparticle_distances(distances, output_dir)
        self._plot_collisions_and_decays(output_dir)
        
        print(f"Standard physics plots saved successfully")
        
    def _plot_positions_vs_time(self, trajectories: Dict, output_dir: str) -> None:
        """Plot X and Y positions of all particles over time."""
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        for particle_id, traj in trajectories.items():
            if len(traj['positions']) > 0:
                x_positions = traj['positions'][:, 0]
                plt.plot(traj['time'], x_positions, label=f'Particle {particle_id[:8]}', 
                        color=traj['color'], alpha=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('X Position (μm)')
        plt.title('X Position vs Time')
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        for particle_id, traj in trajectories.items():
            if len(traj['positions']) > 0:
                y_positions = traj['positions'][:, 1]
                plt.plot(traj['time'], y_positions, label=f'Particle {particle_id[:8]}', 
                        color=traj['color'], alpha=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('Y Position (μm)')
        plt.title('Y Position vs Time')
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/particle_positions_vs_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_velocities_vs_time(self, trajectories: Dict, output_dir: str) -> None:
        """Plot velocity magnitude for all particles over time."""
        plt.figure(figsize=(12, 8))
        
        for particle_id, traj in trajectories.items():
            if len(traj['velocities']) > 0:
                velocity_magnitudes = np.linalg.norm(traj['velocities'], axis=1)
                plt.plot(traj['time'], velocity_magnitudes, 
                        label=f'Particle {particle_id[:8]}', color=traj['color'], alpha=0.8)
                
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity Magnitude (μm/s)')
        plt.title('Velocity vs Time')
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/particle_velocity_vs_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_accelerations_vs_time(self, trajectories: Dict, output_dir: str) -> None:
        """Plot acceleration magnitude for all particles over time."""
        plt.figure(figsize=(12, 8))
        
        for particle_id, traj in trajectories.items():
            if len(traj['accelerations']) > 0:
                acceleration_magnitudes = np.linalg.norm(traj['accelerations'], axis=1)
                plt.plot(traj['time'], acceleration_magnitudes, 
                        label=f'Particle {particle_id[:8]}', color=traj['color'], alpha=0.8)
                
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration Magnitude (μm/s²)')
        plt.title('Acceleration vs Time')
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/particle_acceleration_vs_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_system_energy(self, output_dir: str) -> None:
        """Plot total kinetic energy of the system over time."""
        plt.figure(figsize=(12, 8))
        
        time_data = self.simulation.history['time']
        energy_data = self.simulation.history['total_energy']
        
        plt.plot(time_data, energy_data, 'b-', linewidth=2, label='Total Kinetic Energy')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.title('System Energy vs Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/system_energy_vs_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_interparticle_distances(self, distances: Dict, output_dir: str) -> None:
        """Plot distances between particle pairs over time."""
        if not distances:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Limit plotted pairs to avoid clutter
        max_pairs = 50
        pair_ids = list(distances.keys())
        
        # Select pairs with most interesting behavior (highest variance)
        if len(pair_ids) > max_pairs:
            pair_variances = []
            for pair_id in pair_ids:
                distance_data = distances[pair_id]['distances']
                # Filter out NaN values
                valid_distances = [d for d in distance_data if not np.isnan(d)]
                if len(valid_distances) > 1:
                    variance = np.var(valid_distances)
                    pair_variances.append((pair_id, variance))
                else:
                    pair_variances.append((pair_id, 0))
            
            # Sort by variance and take top pairs
            pair_variances.sort(key=lambda x: x[1], reverse=True)
            selected_pairs = [pair_id for pair_id, _ in pair_variances[:max_pairs]]
            
            print(f"Plotting {len(selected_pairs)} most dynamic pairs out of {len(pair_ids)} total pairs")
        else:
            selected_pairs = pair_ids
            
        # Calculate distance statistics
        all_distances = []
        for pair_id in pair_ids:
            distance_data = distances[pair_id]['distances']
            valid_distances = [d for d in distance_data if not np.isnan(d)]
            all_distances.extend(valid_distances)
        
        if all_distances:
            min_dist = np.min(all_distances)
            max_dist = np.max(all_distances)
            avg_dist = np.mean(all_distances)
            
            print(f"Particle distance analysis:")
            print(f"  Closest approach: {min_dist:.1f} μm")
            print(f"  Maximum separation: {max_dist:.1f} μm")
            print(f"  Average distance: {avg_dist:.1f} μm")
        
        # Use colormap for distinguishing lines
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(selected_pairs)))
        
        # Plot distance over time for each selected pair
        for i, pair_id in enumerate(selected_pairs):
            distance_data = distances[pair_id]
            time_data = distance_data['time']
            dist_data = distance_data['distances']
            
            # Filter out NaN values
            valid_indices = [j for j, d in enumerate(dist_data) if not np.isnan(d)]
            if valid_indices:
                valid_time = [time_data[j] for j in valid_indices]
                valid_dist = [dist_data[j] for j in valid_indices]
                
                plt.plot(valid_time, valid_dist, 
                        color=colors[i], alpha=0.7, linewidth=1.5,
                        label=f'Pair {pair_id}')
                    
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Distance (μm)', fontsize=14)
        plt.title('Interparticle Distances vs Time\n(Most Dynamic Pairs)', fontsize=16, fontweight='bold')
        
        # Adjust legend positioning
        if len(selected_pairs) <= 10:
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=5, fontsize=8)
        else:
            # For many pairs, show legend outside plot area
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/interparticle_distance_hist.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_collisions_and_decays(self, output_dir: str) -> None:
        """Create timeline plot of collision and decay events."""
        plt.figure(figsize=(12, 8))
        
        # Plot collision events as red circles
        collision_times = [event['time'] for event in self.simulation.history['collisions']]
        if collision_times:
            plt.scatter(collision_times, [1] * len(collision_times), 
                       color='red', s=50, label='Collisions', alpha=0.7)
            
        # Plot decay events as black squares
        decay_times = [event['time'] for event in self.simulation.history['decays']]
        if decay_times:
            plt.scatter(decay_times, [0.5] * len(decay_times), 
                       color='black', s=50, label='Decays', alpha=0.7)
            
        # Set x-axis limits to show full simulation duration
        plt.xlim(0, self.simulation.duration)
            
        plt.xlabel('Time (s)')
        plt.ylabel('Event Type')
        plt.title('Collision and Decay Events vs Time')
        plt.yticks([0.5, 1], ['Decays', 'Collisions'])
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/collisions_and_decays_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_summary_plot(self, output_dir: str) -> None:
        """Generate comprehensive summary plot with multiple subplots."""
        # Set up large figure with grid layout
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('Particle Simulation - Comprehensive Analysis Summary', 
                    fontsize=24, fontweight='bold', y=0.95)
        
        # Use GridSpec for organized layout
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        # Get simulation data
        trajectories = self.simulation.get_particle_trajectories()
        time_data = self.simulation.history['time']
        energy_data = self.simulation.history['total_energy']
        
        # Subplot 1: Particle trajectories (spans two columns)
        ax1 = fig.add_subplot(gs[0, :2])
        trajectory_colors = []
        for particle_id, traj in trajectories.items():
            if len(traj['positions']) > 0:
                ax1.plot(traj['positions'][:, 0], traj['positions'][:, 1], 
                       color=traj['color'], alpha=0.8, linewidth=2)
                trajectory_colors.append(traj['color'])
        ax1.set_xlabel('X Position (μm)', fontsize=12)
        ax1.set_ylabel('Y Position (μm)', fontsize=12)
        ax1.set_title('Particle Trajectories', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Display particle statistics on trajectory plot
        if trajectories:
            total_particles = len(trajectories)
            active_particles = sum(1 for traj in trajectories.values() if not traj['decayed'])
            ax1.text(0.02, 0.98, f'Total: {total_particles} particles\nActive: {active_particles} particles', 
                    transform=ax1.transAxes, verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Subplot 2: System energy over time
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(time_data, energy_data, color='#2E86AB', linewidth=3)
        ax2.fill_between(time_data, energy_data, alpha=0.3, color='#2E86AB')
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Energy (J)', fontsize=12)
        ax2.set_title('System Energy', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Display energy conservation statistics
        if energy_data:
            initial_energy = energy_data[0]
            final_energy = energy_data[-1]
            energy_change = ((final_energy - initial_energy) / initial_energy) * 100 if initial_energy != 0 else 0
            ax2.text(0.02, 0.98, f'Initial: {initial_energy:.2e} J\nFinal: {final_energy:.2e} J\nChange: {energy_change:.1f}%', 
                    transform=ax2.transAxes, verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Subplot 3: Active particles over time
        ax3 = fig.add_subplot(gs[0, 3])
        active_particles = self.simulation.history['active_particles']
        ax3.plot(time_data, active_particles, color='#A23B72', linewidth=3)
        ax3.fill_between(time_data, active_particles, alpha=0.3, color='#A23B72')
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.set_ylabel('Active Particles', fontsize=12)
        ax3.set_title('Active Particles', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Velocity distribution over time
        ax4 = fig.add_subplot(gs[1, :2])
        velocity_colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(trajectories)))
        for i, (particle_id, traj) in enumerate(trajectories.items()):
            if len(traj['velocities']) > 0:
                velocity_magnitudes = np.linalg.norm(traj['velocities'], axis=1)
                ax4.plot(traj['time'], velocity_magnitudes, 
                        color=velocity_colors[i], alpha=0.7, linewidth=1.5)
        ax4.set_xlabel('Time (s)', fontsize=12)
        ax4.set_ylabel('Velocity Magnitude (μm/s)', fontsize=12)
        ax4.set_title('Particle Velocities', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Subplot 5: Timeline of collision and decay events
        ax5 = fig.add_subplot(gs[1, 2])
        collision_times = [event['time'] for event in self.simulation.history['collisions']]
        decay_times = [event['time'] for event in self.simulation.history['decays']]
        
        if collision_times:
            ax5.scatter(collision_times, [1] * len(collision_times), 
                       color='#FF6B6B', s=60, alpha=0.8, label='Collisions', marker='o')
        if decay_times:
            ax5.scatter(decay_times, [0.5] * len(decay_times), 
                       color='#4ECDC4', s=60, alpha=0.8, label='Decays', marker='s')
        
        # Set x-axis limits to show full simulation duration
        ax5.set_xlim(0, self.simulation.duration)
        
        ax5.set_xlabel('Time (s)', fontsize=12)
        ax5.set_ylabel('Event Type', fontsize=12)
        ax5.set_title('Collision & Decay Events', fontsize=14, fontweight='bold')
        ax5.set_yticks([0.5, 1])
        ax5.set_yticklabels(['Decays', 'Collisions'])
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # Display event counts
        total_events = len(collision_times) + len(decay_times)
        ax5.text(0.02, 0.98, f'Total Events: {total_events}\nCollisions: {len(collision_times)}\nDecays: {len(decay_times)}', 
                transform=ax5.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Subplot 6: Pie chart of particle types
        ax6 = fig.add_subplot(gs[1, 3])
        particle_types = {}
        
        # Identify particle types by attribute or color
        for particle in self.simulation.particles:
            # Try to get particle type from QuantumParticle first
            if hasattr(particle, 'particle_type'):
                ptype = particle.particle_type
                # Convert internal names to display names
                if ptype == 'quark_up':
                    display_name = 'Up Quark'
                elif ptype == 'quark_down':
                    display_name = 'Down Quark'
                else:
                    display_name = ptype.replace('_', ' ').title()
            else:
                # Fall back to color-based identification
                color = particle.color
                if color == 'cyan' or color == '#00FFFF':
                    display_name = 'Electron'
                elif color == 'magenta' or color == '#FF00FF':
                    display_name = 'Proton'  
                elif color == 'lime' or color == '#32CD32':
                    display_name = 'Neutron'
                elif color == 'gold' or color == '#FFD700':
                    display_name = 'Photon'
                elif color == 'hotpink' or color == '#FF69B4':
                    display_name = 'Up Quark'
                elif color == 'coral' or color == '#FF7F50':
                    display_name = 'Down Quark'
                elif color == 'springgreen':
                    display_name = 'Custom'
                else:
                    display_name = 'Other'
            
            particle_types[display_name] = particle_types.get(display_name, 0) + 1
        
        if particle_types:
            # Define color map for consistent coloring
            color_map = {
                'Electron': '#00FFFF',     # cyan
                'Proton': '#FF00FF',       # magenta
                'Neutron': '#32CD32',      # lime
                'Photon': '#FFD700',       # gold
                'Up Quark': '#FF69B4',     # hotpink
                'Down Quark': '#FF7F50',   # coral
                'Custom': '#00FF7F',       # springgreen
                'Other': '#808080'         # gray
            }
            
            # Create list of colors for particle types
            colors_pie = [color_map.get(ptype, '#808080') for ptype in particle_types.keys()]
            
            # Create pie chart
            pie_result = ax6.pie(list(particle_types.values()), 
                               labels=list(particle_types.keys()), 
                               autopct='%1.1f%%', 
                               colors=colors_pie,
                               startangle=90)
            
            # Improve percentage label formatting
            if len(pie_result) == 3:
                wedges, texts, autotexts = pie_result
                for autotext in autotexts:
                    autotext.set_color('black')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(9)
            
            ax6.set_title('Particle Types Distribution', fontsize=14, fontweight='bold')
            
            # Display detailed particle counts
            total_particles = sum(particle_types.values())
            info_text = f"Total: {total_particles} particles\n"
            for ptype, count in sorted(particle_types.items()):
                info_text += f"{ptype}: {count}\n"
            
            ax6.text(1.3, 0.5, info_text, transform=ax6.transAxes, 
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        else:
            ax6.text(0.5, 0.5, 'No particle data\navailable', 
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
        
        # Subplot 7: Text-based summary of key statistics
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Calculate comprehensive statistics
        total_particles = len(trajectories)
        active_particles = sum(1 for traj in trajectories.values() if not traj['decayed'])
        total_collisions = len(collision_times)
        total_decays = len(decay_times)
        simulation_time = time_data[-1] if time_data else 0
        
        # Calculate average velocity across all particles
        avg_velocities = []
        for traj in trajectories.values():
            if len(traj['velocities']) > 0:
                avg_vel = np.mean([np.linalg.norm(v) for v in traj['velocities']])
                avg_velocities.append(avg_vel)
        
        avg_system_velocity = np.mean(avg_velocities) if avg_velocities else 0
        
        # Format summary statistics
        summary_text = f"""
        SIMULATION SUMMARY STATISTICS
        
        Particles:                    Total: {total_particles}    Active: {active_particles}    Decayed: {total_particles - active_particles}
        Events:                       Collisions: {total_collisions}    Decays: {total_decays}    Total Events: {total_collisions + total_decays}
        Energy:                       Initial: {energy_data[0]:.2e} J    Final: {energy_data[-1]:.2e} J
        Dynamics:                     Simulation Time: {simulation_time:.2f} s    Avg Velocity: {avg_system_velocity:.2e} μm/s
        """
        
        ax7.text(0.5, 0.5, summary_text, transform=ax7.transAxes, 
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.savefig(f"{output_dir}/summary_plot.png", dpi=300, bbox_inches='tight')
        plt.close() 