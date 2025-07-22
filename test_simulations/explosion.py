#!/usr/bin/env python3
"""
Demo: Quantum Explosion Simulation
Particles exploding outward from central point creating spectacular effects.
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
    Demonstrates a particle explosion simulation.
    
    This simulation models an explosive release of various quantum particles
    from a central point, expanding outwards into a uniform magnetic field.
    The full HEP analysis pipeline is used to process the results.
    """
    print("=== Quantum Explosion Simulation ===")
    print("Simulating a spectacular particle explosion in magnetic fields")
    print("This demo shows how particles scatter when exploding outward from a central point")
    print()
    
    # Create dipole magnetic field for complex deflection
    field = MagneticField.create_dipole_field(strength=1.8)
    
    # Create particles in explosion formation
    particles = []
    N_electrons = 50
    N_protons = 50
    N_neutrons = 30
    N_photons = 30
    N_quark_up = 20
    N_quark_down = 20
    R_explosion = 18000  # μm, veliki radijus eksplozije
    V_explosion = 3000   # μm/s, brzina eksplozije

    # Electrons exploding outward
    for i in range(N_electrons):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.1 * R_explosion, 0.3 * R_explosion)  # Unutrašnji radijus
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        # Explosion velocity outward
        vx = V_explosion * np.cos(angle) + np.random.uniform(-500, 500)
        vy = V_explosion * np.sin(angle) + np.random.uniform(-500, 500)
        particles.append(QuantumParticle(
            particle_type='electron',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=30.0
        ))

    # Protons exploding outward
    for i in range(N_protons):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.1 * R_explosion, 0.3 * R_explosion)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vx = V_explosion * 0.8 * np.cos(angle) + np.random.uniform(-400, 400)
        vy = V_explosion * 0.8 * np.sin(angle) + np.random.uniform(-400, 400)
        particles.append(QuantumParticle(
            particle_type='proton',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=30.0
        ))

    # Neutrons exploding outward
    for i in range(N_neutrons):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.1 * R_explosion, 0.3 * R_explosion)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vx = V_explosion * 0.6 * np.cos(angle) + np.random.uniform(-300, 300)
        vy = V_explosion * 0.6 * np.sin(angle) + np.random.uniform(-300, 300)
        particles.append(QuantumParticle(
            particle_type='neutron',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=25.0
        ))

    # Photons exploding outward (fastest)
    for i in range(N_photons):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.1 * R_explosion, 0.3 * R_explosion)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vx = V_explosion * 1.2 * np.cos(angle) + np.random.uniform(-600, 600)
        vy = V_explosion * 1.2 * np.sin(angle) + np.random.uniform(-600, 600)
        particles.append(QuantumParticle(
            particle_type='photon',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=20.0
        ))

    # Up quarks exploding outward
    for i in range(N_quark_up):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.1 * R_explosion, 0.3 * R_explosion)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vx = V_explosion * 0.9 * np.cos(angle) + np.random.uniform(-450, 450)
        vy = V_explosion * 0.9 * np.sin(angle) + np.random.uniform(-450, 450)
        particles.append(QuantumParticle(
            particle_type='quark_up',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=22.0
        ))

    # Down quarks exploding outward
    for i in range(N_quark_down):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.1 * R_explosion, 0.3 * R_explosion)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vx = V_explosion * 0.7 * np.cos(angle) + np.random.uniform(-350, 350)
        vy = V_explosion * 0.7 * np.sin(angle) + np.random.uniform(-350, 350)
        particles.append(QuantumParticle(
            particle_type='quark_down',
            position=np.array([x, y]),
            velocity=np.array([vx, vy]),
            lifetime=28.0
        ))

    # Create a Simulation instance with HEP analysis enabled.
    sim = Simulation(
        particles=particles,
        magnetic_field=field,
        duration=13.0,
        hep_analysis_enabled=True
    )

    print(f"Created {len(particles)} particles for explosion simulation:")
    print(f"  • {N_electrons} electrons (cyan) - exploding outward")
    print(f"  • {N_protons} protons (magenta) - exploding outward")
    print(f"  • {N_neutrons} neutrons (lime) - neutral explosion particles")
    print(f"  • {N_photons} photons (gold) - fast massless explosion particles")
    print(f"  • {N_quark_up} up quarks (hotpink) - fractional charge explosion")
    print(f"  • {N_quark_down} down quarks (coral) - fractional charge explosion")
    print(f"Magnetic field: Dipole field with strength {field.strength} T")
    print(f"Simulation duration: {sim.duration} s")
    print("Expected: Spectacular explosion with magnetic field deflection")
    print("HEP analysis: ENABLED - detector simulation and statistical analysis")
    print()
    
    # Run the simulation.
    print("Running quantum explosion simulation with HEP analysis...")
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
    viz.animate(interval=40)  # Fast animation for explosion effects
    
    # Generate a comprehensive set of static analysis plots.
    print("Generating comprehensive analysis plots with HEP analysis...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_graphs/quantum_explosion_hep_{timestamp}"
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate both standard and HEP-specific visualizations.
    viz.generate_plots(output_dir)
    viz.create_summary_plot(output_dir)
    sim.create_hep_visualizations(output_dir)
    
    print(f"All plots saved to {output_dir}/")
    print()
    print("💥 This simulation demonstrates:")
    print("  • Particle explosion dynamics")
    print("  • Magnetic field deflection effects")
    print("  • Different particle scattering patterns")
    print("  • Charged vs neutral particle behavior")
    print("  • Spectacular multi-particle explosion effects")
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