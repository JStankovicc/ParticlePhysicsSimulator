import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import os
from datetime import datetime
from particle import Particle
from fields import MagneticField

# Import HEP modules
try:
    from detector import HEPDetector
    from analysis import HEPAnalysis
    from plots import HEPVisualizer
    HEP_AVAILABLE = True
except ImportError:
    HEP_AVAILABLE = False
    print("Note: HEP analysis modules not available. Running basic simulation only.")

class Simulation:
    """
    Manages the particle simulation, including the physics loop, event handling,
    and integration with High Energy Physics (HEP) analysis modules.
    """
    
    def __init__(self, particles: List[Particle], magnetic_field: MagneticField,
                 timestep: float = 0.01, duration: float = 10.0,
                 collisions_enabled: bool = True, decay_enabled: bool = True,
                 hep_analysis_enabled: bool = True):
        """
        Initialize simulation with HEP analysis capabilities.
        
        Args:
            particles: List of particles to simulate
            magnetic_field: Magnetic field object
            timestep: Time step in seconds
            duration: Total simulation duration in seconds
            collisions_enabled: Whether to enable collision detection
            decay_enabled: Whether to enable particle decay
            hep_analysis_enabled: Whether to enable HEP analysis
        """
        self.particles = particles
        self.field = magnetic_field
        self.timestep = timestep
        self.duration = duration
        self.collisions_enabled = collisions_enabled
        self.decay_enabled = decay_enabled
        self.hep_analysis_enabled = hep_analysis_enabled and HEP_AVAILABLE
        self.current_time = 0.0
        
        # Data structure to store the simulation's state at each timestep.
        self.history = {
            'time': [],
            'total_energy': [],
            'total_momentum': [],
            'active_particles': [],
            'collisions': [],
            'decays': []
        }
        
        # Initialize HEP components if analysis is enabled.
        if self.hep_analysis_enabled:
            self.hep_detector = HEPDetector()
            self.hep_analysis = HEPAnalysis()
            self.hep_results = {}
        
        # Set the creation time for all initial particles.
        for particle in self.particles:
            particle.creation_time = 0.0
            
    def run(self) -> Dict:
        """
        Run the complete simulation with HEP analysis.
        
        Returns:
            A dictionary containing the simulation results, including HEP analysis if enabled.
        """
        print(f"Setting up simulation with {len(self.particles)} particles")
        print(f"Simulation will run for {self.duration} seconds with {self.timestep}s time steps")
        if self.hep_analysis_enabled:
            print("HEP detector simulation is active - tracking particle interactions")
        
        # The main simulation loop, executed for each timestep.
        time_steps = int(self.duration / self.timestep)
        
        for step in range(time_steps):
            self.current_time = step * self.timestep
            
            # Update particle positions and velocities.
            self._update_particles()
            
            # Simulate particle detection if HEP analysis is active.
            if self.hep_analysis_enabled:
                self._simulate_detector_response()
            
            # Handle collisions between particles.
            if self.collisions_enabled:
                self._detect_and_resolve_collisions()
                
            # Handle particle decays.
            if self.decay_enabled:
                self._check_decays()
                
            # Record the state of the simulation at the current timestep.
            self._store_simulation_state()
            
            # Print a progress update to the console periodically.
            if step % 100 == 0:
                active_count = sum(1 for p in self.particles if not p.decayed)
                print(f"Simulation progress: {self.current_time:.1f}s - {active_count} particles still active")
                
        # Perform the final HEP analysis after the simulation loop completes.
        if self.hep_analysis_enabled:
            self._perform_hep_analysis()
        
        print("Simulation complete! Processing results...")
        return self._get_simulation_results()
    
    def _simulate_detector_response(self) -> None:
        """Simulates the detector's response to each active particle at the current timestep."""
        if not self.hep_analysis_enabled:
            return
        
        for particle in self.particles:
            if not particle.decayed:
                # Each particle has a chance to be detected by the HEP detector.
                hits = self.hep_detector.detect_particle(particle, self.current_time)
    
    def _perform_hep_analysis(self) -> None:
        """Conducts a comprehensive High Energy Physics analysis on the collected detector data."""
        if not self.hep_analysis_enabled:
            return
        
        print("Running HEP analysis on collected data...")
        
        # Reconstruct tracks from detector hits
        tracks = self.hep_detector.reconstruct_tracks()
        
        # Compile data from active particles for spectral analysis.
        particle_data = []
        for particle in self.particles:
            if not particle.decayed and particle.history['energy']:
                particle_info = {
                    'energy': particle.history['energy'][-1],
                    'momentum': np.linalg.norm(particle.get_momentum()),
                    'mass': particle.mass,
                    'charge': particle.charge,
                    'type': getattr(particle, 'particle_type', 'unknown')
                }
                particle_data.append(particle_info)
        
        # Analyze the particle spectrum to find resonances and other phenomena.
        analysis_results = self.hep_analysis.analyze_particle_spectrum(particle_data)
        
        # Calculate the detector's efficiency based on true vs. detected particles.
        detected_particles = []
        for track in tracks:
            detected_particles.append({'type': track['particle_type']})
        
        true_particles = []
        for particle in self.particles:
            true_particles.append({'type': getattr(particle, 'particle_type', 'unknown')})
        
        efficiency_results = self.hep_analysis.calculate_efficiency(detected_particles, true_particles)
        
        # Store all HEP-related results for later use in visualization and reporting.
        self.hep_results = {
            'detector_summary': self.hep_detector.get_detector_summary(),
            'tracks': tracks,
            'analysis_results': analysis_results,
            'efficiency_results': efficiency_results,
            'detector_data': {
                'layers': self.hep_detector.layers,
                'hits': self.hep_detector.hits,
                'tracks': tracks,
                'summary': self.hep_detector.get_detector_summary()
            }
        }
        
        print(f"HEP analysis complete:")
        print(f"  Detector recorded {len(self.hep_detector.hits)} particle hits")
        print(f"  Successfully reconstructed {len(tracks)} particle tracks")
        print(f"  Found {len(analysis_results.get('resonances', []))} potential resonances")
        print(f"  Overall detector efficiency: {efficiency_results.get('overall_efficiency', 0):.1%}")
        
    def create_hep_visualizations(self, output_dir: str) -> None:
        """Generates and saves all HEP-related visualizations."""
        if not self.hep_analysis_enabled or not self.hep_results:
            print("HEP analysis not available - skipping specialized plots")
            return
        
        print("Creating HEP analysis visualizations...")
        
        # Initialize the HEP visualizer with the specified output directory.
        hep_viz = HEPVisualizer(output_dir)
        
        # Create all HEP plots
        hep_viz.plot_detector_layout(self.hep_results['detector_data'], "detector_layout_with_tracks.png")
        hep_viz.plot_invariant_mass_spectrum(self.hep_results['analysis_results'], "invariant_mass_spectrum.png")
        hep_viz.plot_detector_efficiency(self.hep_results['efficiency_results'], "detector_efficiency_by_type.png")
        hep_viz.plot_statistical_analysis(self.hep_results['analysis_results'], "statistical_analysis_overview.png")
        
        # Generate a comprehensive summary report combining simulation and HEP data.
        simulation_data = self._get_simulation_results()
        # Add particles and simulation object reference to the data
        simulation_data['particles'] = self.particles
        simulation_data['simulation'] = self
        hep_viz.create_hep_summary_report(
            simulation_data, 
            self.hep_results['detector_data'], 
            self.hep_results['analysis_results'],
            "hep_analysis_report.png"
        )
        
        print("HEP visualizations saved successfully")
        
    def _update_particles(self) -> None:
        """Updates the state of each active particle for a single timestep."""
        for particle in self.particles:
            if not particle.decayed:
                # The magnetic field affects the particle's acceleration.
                field_vector = self.field.get_field_at(particle.position)
                particle.update(self.timestep, field_vector, self.current_time)
                
    def _detect_and_resolve_collisions(self) -> None:
        """Detects and resolves collisions between all pairs of active particles."""
        collisions_this_step = []
        
        for i, particle1 in enumerate(self.particles):
            if particle1.decayed or not particle1.collidable:
                continue
                
            for j, particle2 in enumerate(self.particles[i+1:], i+1):
                if particle2.decayed or not particle2.collidable:
                    continue
                    
                if particle1.is_colliding_with(particle2):
                    collision_result = self._resolve_collision(particle1, particle2)
                    if collision_result:
                        collisions_this_step.append({
                            'time': self.current_time,
                            'particle1_id': particle1.id,
                            'particle2_id': particle2.id,
                            'type': collision_result['type'],
                            'position': particle1.position.copy()
                        })
                        
        if collisions_this_step:
            self.history['collisions'].extend(collisions_this_step)
            
    def _resolve_collision(self, particle1: Particle, particle2: Particle) -> Optional[Dict]:
        """
        Resolve collision between two particles.
        
        Args:
            particle1: First particle
            particle2: Second particle
            
        Returns:
            A dictionary describing the collision type, or None if no interaction occurs.
        """
        # Check for annihilation (opposite charges)
        if particle1.charge * particle2.charge < 0:
            # Annihilation: both particles decay
            particle1.decayed = True
            particle2.decayed = True
            return {'type': 'annihilation'}
            
        # Elastic collision: exchange momentum
        # Model an elastic collision where momentum and kinetic energy are conserved.
        relative_velocity = particle1.velocity - particle2.velocity
        
        # Calculate collision normal
        collision_normal = particle2.position - particle1.position
        collision_normal = collision_normal / np.linalg.norm(collision_normal)
        
        # Calculate relative velocity along normal
        relative_velocity_normal = np.dot(relative_velocity, collision_normal)
        
        # Only resolve if particles are moving toward each other
        if relative_velocity_normal > 0:
            # Elastic collision formula
            # v1' = v1 - (2*m2/(m1+m2)) * (v1-v2) * dot(n, v1-v2) * n
            # v2' = v2 - (2*m1/(m1+m2)) * (v2-v1) * dot(n, v2-v1) * n
            
            mass1, mass2 = particle1.mass, particle2.mass
            total_mass = mass1 + mass2
            
            # Update velocities
            velocity_change1 = (2 * mass2 / total_mass) * relative_velocity_normal * collision_normal
            velocity_change2 = (2 * mass1 / total_mass) * (-relative_velocity_normal) * collision_normal
            
            particle1.velocity -= velocity_change1
            particle2.velocity += velocity_change2
            
            return {'type': 'elastic'}
            
        return None
        
    def _check_decays(self) -> None:
        """Checks each particle to see if it has decayed during the current timestep."""
        decays_this_step = []
        
        for particle in self.particles:
            if not particle.decayed and particle.check_decay(self.current_time):
                decays_this_step.append({
                    'time': self.current_time,
                    'particle_id': particle.id,
                    'position': particle.position.copy()
                })
                
        if decays_this_step:
            self.history['decays'].extend(decays_this_step)
            
    def _store_simulation_state(self) -> None:
        """Records the current state of the simulation in the history log."""
        active_particles = [p for p in self.particles if not p.decayed]
        
        # Calculate and store summary statistics for the current timestep.
        total_energy = sum(p.get_kinetic_energy() for p in active_particles)
        total_momentum = sum(p.get_momentum() for p in active_particles)
        
        self.history['time'].append(self.current_time)
        self.history['total_energy'].append(total_energy)
        self.history['total_momentum'].append(np.linalg.norm(total_momentum))
        self.history['active_particles'].append(len(active_particles))
        
    def _get_simulation_results(self) -> Dict:
        """Compiles a summary of the simulation results."""
        active_particles = [p for p in self.particles if not p.decayed]
        
        return {
            'final_time': self.current_time,
            'total_particles': len(self.particles),
            'active_particles': self.history['active_particles'][-1] if self.history['active_particles'] else len(active_particles),
            'total_collisions': len(self.history['collisions']),
            'total_decays': len(self.history['decays']),
            'final_energy': self.history['total_energy'][-1] if self.history['total_energy'] else 0,
            'total_events': len(self.history['collisions']) + len(self.history['decays']),
            'time': self.history['time'],
            'history': self.history
        }
        
    def get_particle_trajectories(self) -> Dict:
        """
        Get particle trajectories for plotting.
        
        Returns:
            A dictionary containing the complete trajectory data for each particle.
        """
        trajectories = {}
        
        for particle in self.particles:
            if particle.history['time']:
                trajectories[particle.id] = {
                    'time': particle.history['time'],
                    'positions': np.array(particle.history['position']),
                    'velocities': np.array(particle.history['velocity']),
                    'accelerations': np.array(particle.history['acceleration']),
                    'energies': particle.history['energy'],
                    'color': particle.color,
                    'decayed': particle.decayed
                }
                
        return trajectories
        
    def get_interparticle_distances(self) -> Dict:
        """
        Calculate distances between all particle pairs over time.
        
        Returns:
            A dictionary containing the distance between each pair of particles over time.
        """
        distances = {}
        
        for i, particle1 in enumerate(self.particles):
            for j, particle2 in enumerate(self.particles[i+1:], i+1):
                pair_id = f"{particle1.id[:8]}_{particle2.id[:8]}"
                
                # Calculate the distance for this pair at each recorded timestep.
                pair_distances = []
                for t_idx, time in enumerate(self.history['time']):
                    if (t_idx < len(particle1.history['position']) and 
                        t_idx < len(particle2.history['position'])):
                        pos1 = particle1.history['position'][t_idx]
                        pos2 = particle2.history['position'][t_idx]
                        distance = np.linalg.norm(pos1 - pos2)
                        pair_distances.append(distance)
                    else:
                        pair_distances.append(np.nan)
                        
                distances[pair_id] = {
                    'time': self.history['time'],
                    'distances': pair_distances,
                    'particle1_id': particle1.id,
                    'particle2_id': particle2.id
                }
                
        return distances 