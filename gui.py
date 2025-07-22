import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import List, Dict
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from particles import QuantumParticle
from simulation import Simulation
from fields import MagneticField
from visualizer import Visualizer

class QuantumParticleGUI:
    """
    GUI for quantum particle simulation with interactive setup.
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Quantum Particle Simulator")
        self.root.geometry("900x700")
        
        # Simulation variables
        self.particles: List[QuantumParticle] = []
        self.magnetic_field = None
        self.simulation = None
        self.simulation_counter = 0  # Counter for unique simulation folders
        
        # Load available particle types and add custom option
        self.particle_types = QuantumParticle.get_available_types()
        self.particle_types['custom'] = {
            'name': 'Custom',
            'mass': 1e-30,
            'charge': 0,
            'spin': 0.5,
            'color': 'green',
            'description': 'User-defined particle properties'
        }
        
        self.setup_gui()
        
    def setup_gui(self):
        """Initialize main GUI layout and widgets."""
        # Configure main window and frames
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights for responsive resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Quantum Particle Simulator", 
                               font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 5))
        
        subtitle_label = ttk.Label(main_frame, text="Simulate quantum particles in magnetic fields", 
                                  font=("Arial", 10))
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
         
        # --- Left Panel: Particle Configuration ---
        left_frame = ttk.LabelFrame(main_frame, text="Add Quantum Particles", padding="10")
        left_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 10))
        
        # Particle type selection dropdown
        ttk.Label(left_frame, text="Particle Type:").grid(row=0, column=0, sticky="w", pady=5)
        self.particle_type_var = tk.StringVar(value='electron')
        particle_type_combo = ttk.Combobox(left_frame, textvariable=self.particle_type_var,
                                          values=list(self.particle_types.keys()), state="readonly")
        particle_type_combo.grid(row=0, column=1, sticky="ew", pady=5)
        particle_type_combo.bind('<<ComboboxSelected>>', self.on_particle_type_change)
         
        # Frame for custom particle properties (shown only when 'custom' selected)
        self.custom_frame = ttk.LabelFrame(left_frame, text="Custom Particle Properties", padding="5")
        self.custom_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")
         
        # Widgets for custom particle properties
        ttk.Label(self.custom_frame, text="Name:").grid(row=0, column=0, sticky="w", pady=2)
        self.custom_name_var = tk.StringVar(value="Custom")
        ttk.Entry(self.custom_frame, textvariable=self.custom_name_var, width=20).grid(row=0, column=1, padx=5, pady=2)
         
        ttk.Label(self.custom_frame, text="Mass (kg):").grid(row=1, column=0, sticky="w", pady=2)
        self.custom_mass_var = tk.DoubleVar(value=1e-30)
        ttk.Entry(self.custom_frame, textvariable=self.custom_mass_var, width=20).grid(row=1, column=1, padx=5, pady=2)
         
        ttk.Label(self.custom_frame, text="Charge (C):").grid(row=2, column=0, sticky="w", pady=2)
        self.custom_charge_var = tk.DoubleVar(value=0.0)
        ttk.Entry(self.custom_frame, textvariable=self.custom_charge_var, width=20).grid(row=2, column=1, padx=5, pady=2)
         
        ttk.Label(self.custom_frame, text="Spin:").grid(row=3, column=0, sticky="w", pady=2)
        self.custom_spin_var = tk.DoubleVar(value=0.5)
        ttk.Entry(self.custom_frame, textvariable=self.custom_spin_var, width=20).grid(row=3, column=1, padx=5, pady=2)
         
        ttk.Label(self.custom_frame, text="Color:").grid(row=4, column=0, sticky="w", pady=2)
        self.custom_color_var = tk.StringVar(value="green")
        color_combo = ttk.Combobox(self.custom_frame, textvariable=self.custom_color_var,
                                   values=['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink'],
                                   state="readonly", width=17)
        color_combo.grid(row=4, column=1, padx=5, pady=2)
         
        ttk.Label(self.custom_frame, text="Description:").grid(row=5, column=0, sticky="w", pady=2)
        self.custom_desc_var = tk.StringVar(value="User-defined particle")
        ttk.Entry(self.custom_frame, textvariable=self.custom_desc_var, width=20).grid(row=5, column=1, padx=5, pady=2)
         
        # Custom properties frame hidden by default
        self.custom_frame.grid_remove()
         
        # Text area to display selected particle properties
        self.particle_info_text = tk.Text(left_frame, height=8, width=40)
        self.particle_info_text.grid(row=2, column=0, columnspan=2, pady=10)
         
        # Input fields for particle's initial position
        ttk.Label(left_frame, text="Position (μm):").grid(row=3, column=0, sticky="w", pady=5)
        pos_frame = ttk.Frame(left_frame)
        pos_frame.grid(row=3, column=1, sticky="ew", pady=5)
         
        ttk.Label(pos_frame, text="X:").grid(row=0, column=0)
        self.pos_x_var = tk.DoubleVar(value=0.0)
        ttk.Entry(pos_frame, textvariable=self.pos_x_var, width=10).grid(row=0, column=1, padx=5)
         
        ttk.Label(pos_frame, text="Y:").grid(row=0, column=2, padx=(10, 0))
        self.pos_y_var = tk.DoubleVar(value=0.0)
        ttk.Entry(pos_frame, textvariable=self.pos_y_var, width=10).grid(row=0, column=3, padx=5)
         
        # Input fields for particle's initial velocity
        ttk.Label(left_frame, text="Velocity (μm/s):").grid(row=4, column=0, sticky="w", pady=5)
        vel_frame = ttk.Frame(left_frame)
        vel_frame.grid(row=4, column=1, sticky="ew", pady=5)
         
        ttk.Label(vel_frame, text="Vx:").grid(row=0, column=0)
        self.vel_x_var = tk.DoubleVar(value=1.0)
        ttk.Entry(vel_frame, textvariable=self.vel_x_var, width=10).grid(row=0, column=1, padx=5)
         
        ttk.Label(vel_frame, text="Vy:").grid(row=0, column=2, padx=(10, 0))
        self.vel_y_var = tk.DoubleVar(value=0.0)
        ttk.Entry(vel_frame, textvariable=self.vel_y_var, width=10).grid(row=0, column=3, padx=5)
         
        # Input field for particle's lifetime
        ttk.Label(left_frame, text="Lifetime (s):").grid(row=5, column=0, sticky="w", pady=5)
        self.lifetime_var = tk.DoubleVar(value=10.0)
        ttk.Entry(left_frame, textvariable=self.lifetime_var, width=10).grid(row=5, column=1, sticky="w", pady=5)
         
        # Button to add particle to simulation
        add_button = ttk.Button(left_frame, text="Add Particle", command=self.add_particle)
        add_button.grid(row=6, column=0, columnspan=2, pady=10)
         
        # --- Right Panel: Simulation Controls and Particle List ---
        right_frame = ttk.LabelFrame(main_frame, text="Simulation Controls", padding="10")
        right_frame.grid(row=2, column=1, sticky="nsew")
         
        # Controls for magnetic field configuration
        ttk.Label(right_frame, text="Magnetic Field:").grid(row=0, column=0, sticky="w", pady=5)
         
        self.field_type_var = tk.StringVar(value='uniform')
        field_type_combo = ttk.Combobox(right_frame, textvariable=self.field_type_var,
                                       values=['uniform', 'dipole'], state="readonly")
        field_type_combo.grid(row=0, column=1, sticky="ew", pady=5)
         
        ttk.Label(right_frame, text="Field Strength (T):").grid(row=1, column=0, sticky="w", pady=5)
        self.field_strength_var = tk.DoubleVar(value=1.0)
        ttk.Entry(right_frame, textvariable=self.field_strength_var, width=10).grid(row=1, column=1, sticky="w", pady=5)
         
        # Input for simulation duration
        ttk.Label(right_frame, text="Duration (s):").grid(row=2, column=0, sticky="w", pady=5)
        self.duration_var = tk.DoubleVar(value=8.0)
        ttk.Entry(right_frame, textvariable=self.duration_var, width=10).grid(row=2, column=1, sticky="w", pady=5)
         
        # Listbox to display added particles
        ttk.Label(right_frame, text="Added Particles:").grid(row=3, column=0, sticky="w", pady=(20, 5))
         
        self.particle_listbox = tk.Listbox(right_frame, height=8)
        self.particle_listbox.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
         
        # Buttons for managing particle list
        remove_button = ttk.Button(right_frame, text="Remove Selected", command=self.remove_particle)
        remove_button.grid(row=5, column=0, columnspan=2, pady=5)
         
        clear_button = ttk.Button(right_frame, text="Clear All", command=self.clear_particles)
        clear_button.grid(row=6, column=0, columnspan=2, pady=5)
         
        # Button to start simulation
        run_button = ttk.Button(right_frame, text="Run Simulation", command=self.run_simulation)
        run_button.grid(row=7, column=0, columnspan=2, pady=20)
         
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Add quantum particles to start simulation")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_label.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
         
        # Initialize particle info display
        self.on_particle_type_change()
         
    def on_particle_type_change(self, event=None):
        """Handle particle type dropdown selection changes."""
        particle_type = self.particle_type_var.get()
         
        # Show/hide custom properties frame
        if particle_type == 'custom':
            self.custom_frame.grid()
            # Display current custom values
            info_text = f"Name: {self.custom_name_var.get()}\n"
            info_text += f"Mass: {self.custom_mass_var.get():.2e} kg\n"
            info_text += f"Charge: {self.custom_charge_var.get():.2e} C\n"
            info_text += f"Spin: {self.custom_spin_var.get()}\n"
            info_text += f"Color: {self.custom_color_var.get()}\n"
            info_text += f"Description: {self.custom_desc_var.get()}"
        elif particle_type in self.particle_types:
            self.custom_frame.grid_remove()
            # Display predefined particle properties
            info = self.particle_types[particle_type]
             
            info_text = f"Name: {info['name']}\n"
            info_text += f"Mass: {info['mass']:.2e} kg\n"
            info_text += f"Charge: {info['charge']:.2e} C\n"
            info_text += f"Spin: {info['spin']}\n"
            info_text += f"Color: {info['color']}\n"
            info_text += f"Description: {info['description']}"
        
        self.particle_info_text.delete(1.0, tk.END)
        self.particle_info_text.insert(1.0, info_text)
             
    def add_particle(self):
        """Add new particle to simulation based on GUI settings."""
        try:
            particle_type = self.particle_type_var.get()
            position = np.array([self.pos_x_var.get(), self.pos_y_var.get()])
            velocity = np.array([self.vel_x_var.get(), self.vel_y_var.get()])
            lifetime = self.lifetime_var.get()
             
            if particle_type == 'custom':
                # Create custom particle with input field values
                custom_properties = {
                    'mass': self.custom_mass_var.get(),
                    'charge': self.custom_charge_var.get(),
                    'spin': self.custom_spin_var.get(),
                    'color': self.custom_color_var.get(),
                    'name': self.custom_name_var.get(),
                    'description': self.custom_desc_var.get()
                }
                particle = QuantumParticle('custom', position, velocity, lifetime, custom_properties)
            else:
                particle = QuantumParticle(particle_type, position, velocity, lifetime)
            
            self.particles.append(particle)
             
            # Update particle listbox
            self.update_particle_list()
             
            if particle_type == 'custom':
                self.status_var.set(f"Added custom particle: {self.custom_name_var.get()}")
            else:
                self.status_var.set(f"Added {particle_type} particle")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add particle: {str(e)}")
             
    def remove_particle(self):
        """Remove selected particle from simulation list."""
        selection = self.particle_listbox.curselection()
        if selection:
            index = selection[0]
            removed_particle = self.particles.pop(index)
            self.update_particle_list()
            self.status_var.set(f"Removed {removed_particle.particle_type} particle")
             
    def clear_particles(self):
        """Remove all particles from simulation list."""
        self.particles.clear()
        self.update_particle_list()
        self.status_var.set("Cleared all particles")
         
    def update_particle_list(self):
        """Refresh listbox to show current particle list."""
        self.particle_listbox.delete(0, tk.END)
        for i, particle in enumerate(self.particles):
            self.particle_listbox.insert(tk.END, 
                f"{i+1}. {particle.particle_type} at ({particle.position[0]:.1f}, {particle.position[1]:.1f})")
             
    def run_simulation(self):
        """Run particle simulation with current settings."""
        if not self.particles:
            messagebox.showwarning("Warning", "No particles added to simulation!")
            return
            
        try:
            # Create magnetic field
            field_type = self.field_type_var.get()
            strength = self.field_strength_var.get()
            
            if field_type == 'uniform':
                self.magnetic_field = MagneticField.create_uniform_field(strength)
            elif field_type == 'dipole':
                self.magnetic_field = MagneticField.create_dipole_field(strength)
            else:
                raise ValueError(f"Unknown field type: {field_type}")
                 
            # Initialize simulation with particles and field settings
            self.simulation = Simulation(
                particles=list(self.particles),  # Cast to list of base Particle type
                magnetic_field=self.magnetic_field,
                duration=self.duration_var.get(),
                hep_analysis_enabled=True  # Always enable HEP analysis
            )
            
            self.status_var.set("Running simulation with HEP analysis...")
            self.root.update()
             
            # Execute simulation
            results = self.simulation.run()
             
            # Prepare for visualization
            self.status_var.set("Creating particle visualization...")
            self.root.update()
             
            viz = Visualizer(self.simulation)
             
            # Set up unique output directory
            self.status_var.set("Generating analysis plots...")
            self.root.update()
             
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.simulation_counter += 1
            output_dir = f"output_graphs/quantum_hep_simulation_{timestamp}_run_{self.simulation_counter:03d}"
             
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
             
            # Generate standard physics plots
            viz.generate_plots(output_dir)
             
            # Generate HEP visualizations
            self.status_var.set("Creating HEP analysis visualizations...")
            self.root.update()
             
            self.simulation.create_hep_visualizations(output_dir)
             
            # Compile summary message with results
            hep_results = getattr(self.simulation, 'hep_results', {})
            if hep_results:
                detector_summary = hep_results.get('detector_summary', {})
                analysis_results = hep_results.get('analysis_results', {})
                
                message = f"""Quantum Particle Simulation with HEP Analysis Completed!

Standard Results:
• Total particles: {results['total_particles']}
• Active particles: {results['active_particles']}
• Total collisions: {results['total_collisions']}
• Final energy: {results['final_energy']:.2e} J

HEP Analysis Results:
• Detector hits: {detector_summary.get('total_hits', 0)}
• Reconstructed tracks: {detector_summary.get('reconstructed_tracks', 0)}
• Detector efficiency: {detector_summary.get('detector_efficiency', 0):.3f}
• Resonances found: {len(analysis_results.get('resonances', []))}
• Analysis quality: {analysis_results.get('analysis_quality', 'N/A')}

Visualizations saved to: {output_dir}

HEP-specific plots generated:
• Detector layout with particle tracks
• Invariant mass spectrum with resonances
• Detector efficiency analysis
• Statistical significance analysis
• Comprehensive HEP analysis report"""
            else:
                message = f"""Quantum Particle Simulation Completed!

Results:
• Total particles: {results['total_particles']}
• Active particles: {results['active_particles']}
• Total collisions: {results['total_collisions']}
• Final energy: {results['final_energy']:.2e} J

Visualizations saved to: {output_dir}

Note: HEP analysis was not available for this simulation."""
             
            # Display results and animation
            try:
                if self.root.winfo_exists():
                    self.status_var.set("Simulation completed successfully!")
                    self.root.update()
                     
                    # Display animation after plots are generated
                    self.status_var.set("Starting animation...")
                    self.root.update()
                     
                    # Run animation in non-blocking manner
                    viz.animate(interval=50)
                     
                    # Check if GUI still exists after animation
                    if self.root.winfo_exists():
                        messagebox.showinfo("Simulation Complete", message)
                        self.status_var.set("Simulation completed successfully!")
                else:
                    print("GUI was closed during the simulation.")
                    print(message)  # Print to console instead
            except tk.TclError:
                print("GUI was closed during the simulation.")
                print(message)  # Print to console instead
             
        except Exception as e:
            error_msg = f"An error occurred during the simulation: {str(e)}"
             
            # Check if GUI still exists before showing error
            try:
                if self.root.winfo_exists():
                    messagebox.showerror("Error", error_msg)
                    self.status_var.set(f"Error: {str(e)}")
                else:
                    print(error_msg)
            except tk.TclError:
                print(error_msg)
             
            import traceback
            traceback.print_exc()
             
    def run(self):
        """Start Tkinter main event loop."""
        self.root.mainloop()

def main():
    """Main function to run the quantum particle GUI."""
    app = QuantumParticleGUI()
    app.run()

if __name__ == "__main__":
    main() 