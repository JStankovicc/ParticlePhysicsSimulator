#!/usr/bin/env python3
"""
Demo: Quantum Tornado Simulation
Particles moving in spiral patterns creating tornado-like effects in magnetic fields.
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
    Demonstrates a simulation of particles forming a tornado-like vortex.
    
    This simulation initializes particles with tangential velocities, causing them
    to spiral within a uniform magnetic field, mimicking the appearance of a tornado.
    The full HEP analysis pipeline is used to process the results.
    """
    print("=== Quantum Tornado Simulation ===")
    print("Creating spiral particle motion that looks like a tornado")
    print("This demo shows how magnetic fields can create beautiful spiral patterns")
    print()
    
    # Create dipole magnetic field for spiral effects
    field = MagneticField.create_dipole_field(strength=2.0)
    
    # Create particles in spiral formation
    particles = []
    N_electrons = 60
    N_protons = 60
    N_neutrons = 20
    N_photons = 20
    N_quark_up = 20
    N_quark_down = 20
    R_tornado = 18000  # μm, veliki radijus tornada
    V_tornado = 2000   # μm/s, brzina tornada

    # Inner spiral: Electrons moving clockwise
    for i in range(N_electrons):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.1 * R_tornado, 0.5 * R_tornado)  # Unutrašnji spiral
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        # Tangential velocity for spiral motion (clockwise)
        vx = -V_tornado * np.sin(angle) + np.random.uniform(-300, 300)
        vy = V_tornado * np.cos(angle) + np.random.uniform(-300, 300)
        particles.append(QuantumParticle(
            particle_type='electron',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=25.0
        ))

    # Outer spiral: Protons moving counter-clockwise
    for i in range(N_protons):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.5 * R_tornado, R_tornado)  # Spoljašnji spiral
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        # Opposite tangential velocity (counter-clockwise)
        vx = V_tornado * 0.8 * np.sin(angle) + np.random.uniform(-250, 250)
        vy = -V_tornado * 0.8 * np.cos(angle) + np.random.uniform(-250, 250)
        particles.append(QuantumParticle(
            particle_type='proton',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=25.0
        ))

    # Central particles: Neutrons
    for i in range(N_neutrons):
        x = np.random.uniform(-0.3 * R_tornado, 0.3 * R_tornado)
        y = np.random.uniform(-0.3 * R_tornado, 0.3 * R_tornado)
        vx = np.random.uniform(-V_tornado * 0.5, V_tornado * 0.5)
        vy = np.random.uniform(-V_tornado * 0.5, V_tornado * 0.5)
        particles.append(QuantumParticle(
            particle_type='neutron',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=20.0
        ))

    # Central particles: Photons
    for i in range(N_photons):
        x = np.random.uniform(-0.3 * R_tornado, 0.3 * R_tornado)
        y = np.random.uniform(-0.3 * R_tornado, 0.3 * R_tornado)
        vx = np.random.uniform(-V_tornado * 0.7, V_tornado * 0.7)
        vy = np.random.uniform(-V_tornado * 0.7, V_tornado * 0.7)
        particles.append(QuantumParticle(
            particle_type='photon',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=18.0
        ))

    # Intermediate orbits: Up quarks
    for i in range(N_quark_up):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.3 * R_tornado, 0.7 * R_tornado)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vx = np.random.uniform(-V_tornado * 0.6, V_tornado * 0.6)
        vy = np.random.uniform(-V_tornado * 0.6, V_tornado * 0.6)
        particles.append(QuantumParticle(
            particle_type='quark_up',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=22.0
        ))

    # Intermediate orbits: Down quarks
    for i in range(N_quark_down):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.3 * R_tornado, 0.7 * R_tornado)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vx = np.random.uniform(-V_tornado * 0.6, V_tornado * 0.6)
        vy = np.random.uniform(-V_tornado * 0.6, V_tornado * 0.6)
        particles.append(QuantumParticle(
            particle_type='quark_down',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=22.0
        ))

    # Create a Simulation instance with HEP analysis enabled.
    sim = Simulation(
        particles=particles,
        magnetic_field=field,
        duration=15.0,
        hep_analysis_enabled=True
    )

    print(f"Created {len(particles)} particles for tornado simulation:")
    print(f"  • {N_electrons} electrons (cyan) - inner clockwise spiral")
    print(f"  • {N_protons} protons (magenta) - outer counter-clockwise spiral")
    print(f"  • {N_neutrons} neutrons (lime) - central particles")
    print(f"  • {N_photons} photons (gold) - central massless particles")
    print(f"  • {N_quark_up} up quarks (hotpink) - intermediate orbits")
    print(f"  • {N_quark_down} down quarks (coral) - intermediate orbits")
    print(f"Magnetic field: Dipole field with strength {field.strength} T")
    print(f"Simulation duration: {sim.duration} s")
    print("Expected: Spiral tornado-like particle motion and vortex formation")
    print("HEP analysis: ENABLED - detector simulation and statistical analysis")
    print()
    
    # Run the simulation.
    print("Running quantum tornado simulation with HEP analysis...")
    results = sim.run()
    
    # Display a summary of the primary simulation results.
    print(f"Simulation completed!")
    print(f"Total particles: {results['total_particles']}")
    print(f"Active particles: {results['active_particles']}")
    print(f"Collisions detected: {results['total_collisions']}")
    print(f"Particle decays: {results['total_decays']}")
    print(f"Final energy: {results['final_energy']:.2e} J")
    
    # Show HEP results
    if hasattr(sim, 'hep_results') and sim.hep_results:
        hep_results = sim.hep_results
        detector_summary = hep_results.get('detector_summary', {})
        analysis_results = hep_results.get('analysis_results', {})
        
        print(f"\n=== HEP Analysis Results ===")
        print(f"Detector hits: {detector_summary.get('total_hits', 0)}")
        print(f"Reconstructed tracks: {detector_summary.get('reconstructed_tracks', 0)}")
        print(f"Detector efficiency: {detector_summary.get('detector_efficiency', 0):.3f}")
        print(f"Resonances found: {len(analysis_results.get('resonances', []))}")
        print(f"Analysis quality: {analysis_results.get('analysis_quality', 'N/A')}")
    print()
    
    # Create an animated visualization.
    print("Creating advanced animation...")
    from visualizer import Visualizer
    viz = Visualizer(sim)
    viz.animate(interval=40)  # Medium speed for spiral effects
    
    # Generate a comprehensive set of static analysis plots.
    print("Generating comprehensive analysis plots with HEP analysis...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_graphs/quantum_tornado_hep_{timestamp}"
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate both standard and HEP-specific visualizations.
    viz.generate_plots(output_dir)
    viz.create_summary_plot(output_dir)
    sim.create_hep_visualizations(output_dir)
    
    print(f"All plots saved to {output_dir}/")
    print()
    print("🌪️ This simulation demonstrates:")
    print("  • Spiral vortex formation")
    print("  • Charged particle orbital mechanics")
    print("  • Magnetic field gradient effects")
    print("  • Particle separation by charge")
    print("  • Complex multi-particle dynamics")
    print("  • HEP detector simulation and analysis")
    print("  • Statistical significance of resonances")
    print("  • Professional scientific visualization")
    print()
    print("📊 Generated visualizations:")
    print("  • Standard particle physics plots")
    print("  • HEP detector layout with particle tracks")
    print("  • Invariant mass spectrum with resonances")
    print("  • Detector efficiency analysis")
    print("  • Statistical significance analysis")
    print("  • Comprehensive HEP analysis report")

if __name__ == "__main__":
    main() 