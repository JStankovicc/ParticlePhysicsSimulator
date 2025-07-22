#!/usr/bin/env python3
"""
Demo: Quantum Particle Collider Simulation
Advanced simulation demonstrating particle collisions and interactions in complex magnetic fields.
"""

import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import Simulation
from fields import MagneticField
from particles import QuantumParticle
import numpy as np

def main():
    """
    Demonstrates a high-energy particle collision experiment.
    
    This simulation models two opposing particle beams (electrons and protons)
    colliding in a dipole magnetic field, along with a cloud of other particle types.
    It leverages the full HEP analysis pipeline to process the results.
    """
    print("=== Quantum Particle Collider Simulation ===")
    print("Simulating high-energy particle collisions in magnetic fields")
    print("This demo shows how different particles interact when they collide")
    print()
    
    # Create complex dipole magnetic field for interesting trajectories
    field = MagneticField.create_dipole_field(strength=2.5)
    
    # Create particles for collision experiments
    particles = []
    N_electrons = 40
    N_protons = 40
    N_neutrons = 15
    N_photons = 15
    N_quark_up = 20
    N_quark_down = 20
    R_collider = 18000  # μm, veliki radijus collider-a
    V_beam = 2500      # μm/s, brzina beam-a

    # Particle beam 1: Electrons moving from left - focused collision zone
    for i in range(N_electrons):
        x = np.random.uniform(-R_collider, -0.9 * R_collider)
        y = np.random.uniform(-0.2 * R_collider, 0.2 * R_collider)  # Tighter beam
        vx = V_beam + np.random.uniform(-100, 100)  # More focused velocity
        vy = np.random.uniform(-200, 200)  # Reduced spread
        particles.append(QuantumParticle(
            particle_type='electron',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=20.0
        ))

    # Particle beam 2: Protons moving from right - focused collision zone
    for i in range(N_protons):
        x = np.random.uniform(0.9 * R_collider, R_collider)
        y = np.random.uniform(-0.2 * R_collider, 0.2 * R_collider)  # Tighter beam
        vx = -V_beam + np.random.uniform(-100, 100)  # More focused velocity
        vy = np.random.uniform(-200, 200)  # Reduced spread
        particles.append(QuantumParticle(
            particle_type='proton',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=20.0
        ))

    # Neutral particles: Neutrons in collision zone
    for i in range(N_neutrons):
        x = np.random.uniform(-0.3 * R_collider, 0.3 * R_collider)  # Central collision zone
        y = np.random.uniform(-0.3 * R_collider, 0.3 * R_collider)
        vx = np.random.uniform(-V_beam * 0.8, V_beam * 0.8)  # Higher velocities for more collisions
        vy = np.random.uniform(-V_beam * 0.8, V_beam * 0.8)
        particles.append(QuantumParticle(
            particle_type='neutron',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=18.0
        ))

    # Massless particles: Photons in collision zone
    for i in range(N_photons):
        x = np.random.uniform(-0.3 * R_collider, 0.3 * R_collider)  # Central collision zone
        y = np.random.uniform(-0.3 * R_collider, 0.3 * R_collider)
        vx = np.random.uniform(-V_beam * 1.0, V_beam * 1.0)  # Higher velocities
        vy = np.random.uniform(-V_beam * 1.0, V_beam * 1.0)
        particles.append(QuantumParticle(
            particle_type='photon',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=15.0
        ))

    # Quark particles: Up quarks in collision zone
    for i in range(N_quark_up):
        x = np.random.uniform(-0.4 * R_collider, 0.4 * R_collider)  # Wider collision zone
        y = np.random.uniform(-0.4 * R_collider, 0.4 * R_collider)
        vx = np.random.uniform(-V_beam * 0.9, V_beam * 0.9)  # Higher velocities
        vy = np.random.uniform(-V_beam * 0.9, V_beam * 0.9)
        particles.append(QuantumParticle(
            particle_type='quark_up',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=16.0
        ))

    # Quark particles: Down quarks in collision zone
    for i in range(N_quark_down):
        x = np.random.uniform(-0.4 * R_collider, 0.4 * R_collider)  # Wider collision zone
        y = np.random.uniform(-0.4 * R_collider, 0.4 * R_collider)
        vx = np.random.uniform(-V_beam * 0.9, V_beam * 0.9)  # Higher velocities
        vy = np.random.uniform(-V_beam * 0.9, V_beam * 0.9)
        particles.append(QuantumParticle(
            particle_type='quark_down',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=16.0
        ))

    # Create a Simulation instance with HEP analysis enabled.
    sim = Simulation(
        particles=particles,
        magnetic_field=field,
        duration=12.0,
        hep_analysis_enabled=True
    )

    print(f"Setting up particle beams:")
    print(f"  {N_electrons} electrons moving from left side")
    print(f"  {N_protons} protons moving from right side") 
    print(f"  {N_neutrons} neutrons scattered around collision zone")
    print(f"  {N_photons} photons (massless) in central area")
    print(f"  {N_quark_up} up quarks with fractional positive charge")
    print(f"  {N_quark_down} down quarks with fractional negative charge")
    print(f"Magnetic field: Dipole field at {field.strength} Tesla")
    print(f"Collision zone: {R_collider} μm radius")
    print("Expected result: Complex collision patterns and particle interactions")
    print("HEP analysis will track all particle interactions through detector layers")
    print()
    
    # Run the simulation.
    print("Starting collision simulation...")
    results = sim.run()
    
    # Display summary of primary simulation results.
    print()
    print("=== Simulation Results ===")
    print(f"Total particles simulated: {results['total_particles']}")
    print(f"Particles still active: {results['active_particles']}")
    print(f"Collisions detected: {results['total_collisions']}")
    print(f"Particle decays observed: {results['total_decays']}")
    print(f"Final system energy: {results['final_energy']:.2e} Joules")
    
    # Show HEP results
    if hasattr(sim, 'hep_results') and sim.hep_results:
        hep_results = sim.hep_results
        detector_summary = hep_results.get('detector_summary', {})
        analysis_results = hep_results.get('analysis_results', {})
        
        print()
        print("=== HEP Detector Analysis ===")
        print(f"Detector hits recorded: {detector_summary.get('total_hits', 0)}")
        print(f"Particle tracks reconstructed: {detector_summary.get('reconstructed_tracks', 0)}")
        print(f"Detector efficiency: {detector_summary.get('detector_efficiency', 0):.1%}")
        print(f"Resonances discovered: {len(analysis_results.get('resonances', []))}")
        print(f"Analysis quality: {analysis_results.get('analysis_quality', 'N/A')}")
    print()
    
    # Create an animated visualization of the simulation.
    print("Creating particle animation - you'll see the collision in real time")
    from visualizer import Visualizer
    viz = Visualizer(sim)
    viz.animate(interval=40)
    
    # Generate a comprehensive set of static plots.
    print("Generating analysis plots and saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_graphs/quantum_collider_hep_{timestamp}"
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate both standard and HEP-specific visualizations.
    viz.generate_plots(output_dir)
    viz.create_summary_plot(output_dir)
    sim.create_hep_visualizations(output_dir)
    
    print()
    print("=== Results Summary ===")
    print("This collision simulation demonstrated:")
    print("  - Particle beam interactions in magnetic fields")
    print("  - Different behaviors of charged vs neutral particles")
    print("  - Quark dynamics with fractional charges")
    print("  - Professional HEP detector simulation")
    print("  - Statistical analysis of particle resonances")
    print()
    print(f"All analysis plots and data saved to:")
    print(f"  {output_dir}/")
    print()
    print("Generated visualizations include:")
    print("  - Particle trajectory plots")
    print("  - Energy and collision analysis")
    print("  - HEP detector layout with particle tracks")
    print("  - Invariant mass spectrum analysis")
    print("  - Detector efficiency measurements")
    print("  - Statistical significance analysis")
    print("  - Comprehensive scientific report")

if __name__ == "__main__":
    main() 