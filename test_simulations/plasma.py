#!/usr/bin/env python3
"""
Demo: Quantum Plasma Simulation
High-density particle simulation demonstrating plasma-like behavior in magnetic fields.
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
    Demonstrates a high-density plasma simulation.
    
    This simulation models a cloud of various quantum particles confined by a strong
    uniform magnetic field, exhibiting plasma-like collective behavior. The full
    HEP analysis pipeline is used to analyze the detector response.
    """
    print("=== Quantum Plasma Simulation ===")
    print("Simulating high-density particle interactions in strong magnetic fields")
    print("This demo shows plasma-like behavior with particle confinement")
    print()
    
    # Create strong uniform magnetic field for plasma confinement
    field = MagneticField.create_uniform_field(strength=3.0)
    
    # Create dense particle plasma
    particles = []
    N_electrons = 80
    N_protons = 80
    N_neutrons = 10
    N_photons = 10
    N_quark_up = 10
    N_quark_down = 10
    R_cloud = 18000  # μm, veliki oblak
    V_max = 2500     # μm/s, maksimalna brzina

    # Electron cloud (negative charges)
    for i in range(N_electrons):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.2 * R_cloud, R_cloud)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vx = np.random.uniform(-V_max, V_max)
        vy = np.random.uniform(-V_max, V_max)
        particles.append(QuantumParticle(
            particle_type='electron',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=25.0
        ))

    # Proton cloud (positive charges)
    for i in range(N_protons):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.2 * R_cloud, R_cloud)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vx = np.random.uniform(-V_max, V_max)
        vy = np.random.uniform(-V_max, V_max)
        particles.append(QuantumParticle(
            particle_type='proton',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=25.0
        ))

    # Neutrons
    for i in range(N_neutrons):
        x = np.random.uniform(-R_cloud, R_cloud)
        y = np.random.uniform(-R_cloud, R_cloud)
        vx = np.random.uniform(-V_max, V_max)
        vy = np.random.uniform(-V_max, V_max)
        particles.append(QuantumParticle(
            particle_type='neutron',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=20.0
        ))

    # Photons
    for i in range(N_photons):
        x = np.random.uniform(-R_cloud, R_cloud)
        y = np.random.uniform(-R_cloud, R_cloud)
        vx = np.random.uniform(-V_max, V_max)
        vy = np.random.uniform(-V_max, V_max)
        particles.append(QuantumParticle(
            particle_type='photon',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=18.0
        ))

    # Up quarks
    for i in range(N_quark_up):
        x = np.random.uniform(-R_cloud, R_cloud)
        y = np.random.uniform(-R_cloud, R_cloud)
        vx = np.random.uniform(-V_max, V_max)
        vy = np.random.uniform(-V_max, V_max)
        particles.append(QuantumParticle(
            particle_type='quark_up',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=15.0
        ))

    # Down quarks
    for i in range(N_quark_down):
        x = np.random.uniform(-R_cloud, R_cloud)
        y = np.random.uniform(-R_cloud, R_cloud)
        vx = np.random.uniform(-V_max, V_max)
        vy = np.random.uniform(-V_max, V_max)
        particles.append(QuantumParticle(
            particle_type='quark_down',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=15.0
        ))

    # Create a Simulation instance with HEP analysis enabled.
    sim = Simulation(
        particles=particles,
        magnetic_field=field,
        duration=14.0,
        hep_analysis_enabled=True
    )

    print(f"Setting up dense particle cloud:")
    print(f"  {N_electrons} electrons (negative charge cloud)")
    print(f"  {N_protons} protons (positive charge cloud)")
    print(f"  {N_neutrons} neutrons (neutral particles)")
    print(f"  {N_photons} photons (massless, travel at light speed)")
    print(f"  {N_quark_up} up quarks (fractional positive charge)")
    print(f"  {N_quark_down} down quarks (fractional negative charge)")
    print(f"Magnetic field: Strong uniform field at {field.strength} Tesla for confinement")
    print(f"Plasma cloud radius: {R_cloud} μm")
    print("Expected result: Collective plasma behavior and magnetic confinement")
    print("HEP analysis will study particle interactions in dense environment")
    print()
    
    # Run the simulation.
    print("Starting plasma simulation...")
    results = sim.run()
    
    # Display a summary of the primary simulation results.
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
    
    # Create an animated visualization.
    print("Creating plasma animation - watch the particles interact in the magnetic field")
    from visualizer import Visualizer
    viz = Visualizer(sim)
    viz.animate(interval=40)  # Fast animation for many particles
    
    # Generate a comprehensive set of static analysis plots.
    print("Generating analysis plots and saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_graphs/quantum_plasma_hep_{timestamp}"
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate both standard and HEP-specific visualizations.
    viz.generate_plots(output_dir)
    viz.create_summary_plot(output_dir)
    sim.create_hep_visualizations(output_dir)
    
    print()
    print("=== Results Summary ===")
    print("This plasma simulation demonstrated:")
    print("  - High-density particle interactions")
    print("  - Magnetic confinement effects")
    print("  - Charge separation in plasma")
    print("  - Collective particle behavior")
    print("  - Dense environment detector response")
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