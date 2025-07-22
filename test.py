#!/usr/bin/env python3
"""
Test script for Particle Simulation in Magnetic Field
"""

import sys
import os
import numpy as np
from particles import QuantumParticle
from simulation import Simulation
from fields import MagneticField
from visualizer import Visualizer

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from particle import Particle
        print("particle.py imported successfully")
    except ImportError as e:
        print(f"Failed to import particle.py: {e}")
        return False
        
    try:
        from fields import MagneticField
        print("magnetic_field.py imported successfully")
    except ImportError as e:
        print(f"Failed to import magnetic_field.py: {e}")
        return False
        
    try:
        from simulation import Simulation
        print("simulation.py imported successfully")
    except ImportError as e:
        print(f"Failed to import simulation.py: {e}")
        return False
        
    try:
        from visualizer import Visualizer
        print("visualizer.py imported successfully")
    except ImportError as e:
        print(f"Failed to import visualizer.py: {e}")
        return False
        
    return True

def calculate_cyclotron_frequency(charge: float, mass: float, magnetic_field: float) -> float:
    """
    Calculates the cyclotron frequency for a charged particle in a magnetic field.
    
    Args:
        charge: Particle charge in C
        mass: Particle mass in kg
        magnetic_field: Magnetic field strength in T
        
    Returns:
        Cyclotron frequency in Hz
    """
    return abs(charge * magnetic_field) / (2 * np.pi * mass)

def calculate_larmor_radius(velocity: float, charge: float, mass: float, magnetic_field: float) -> float:
    """
    Calculates the Larmor radius for a charged particle in a magnetic field.
    
    Args:
        velocity: Particle velocity in m/s
        charge: Particle charge in C
        mass: Particle mass in kg
        magnetic_field: Magnetic field strength in T
        
    Returns:
        Larmor radius in m
    """
    return mass * velocity / (abs(charge) * magnetic_field)

def test_particle_creation():
    """Test particle creation and basic functionality."""
    print("\nTesting particle creation...")
    
    try:
        from particle import Particle
        
        # Create a test particle
        particle = Particle(
            mass=1e-27,
            charge=1.6e-19,
            position=np.array([0.0, 0.0]),
            velocity=np.array([1000.0, 0.0]),
            lifetime=None,
            color='red',
            radius=0.1
        )
        
        print(f"Particle created: {particle}")
        print(f"  Mass: {particle.mass:.2e} kg")
        print(f"  Charge: {particle.charge:.2e} C")
        print(f"  Position: {particle.position}")
        print(f"  Velocity: {particle.velocity}")
        print(f"  Kinetic Energy: {particle.get_kinetic_energy():.2e} J")
        
        return True
        
    except Exception as e:
        print(f"Particle creation failed: {e}")
        return False

def test_magnetic_field():
    """Test magnetic field creation and functionality."""
    print("\nTesting magnetic field...")
    
    try:
        from fields import MagneticField
        
        # Test uniform field
        uniform_field = MagneticField.create_uniform_field(1.0)
        field_at_origin = uniform_field.get_field_at(np.array([0.0, 0.0]))
        print(f"Uniform field created: {uniform_field}")
        print(f"  Field at origin: {field_at_origin}")
        
        # Test non-uniform field
        nonuniform_field = MagneticField.create_nonuniform_field()
        field_at_point = nonuniform_field.get_field_at(np.array([1.0, 1.0]))
        print(f"Non-uniform field created: {nonuniform_field}")
        print(f"  Field at (1,1): {field_at_point}")
        
        return True
        
    except Exception as e:
        print(f"Magnetic field test failed: {e}")
        return False

def test_simulation():
    """Test simulation creation and basic run."""
    print("\nTesting simulation...")
    
    try:
        from particle import Particle
        from fields import MagneticField
        from simulation import Simulation
        
        # Create test particles
        particles = [
            Particle(
                mass=1e-27,
                charge=1.6e-19,
                position=np.array([0.0, 0.0]),
                velocity=np.array([1000.0, 0.0]),
                color='red'
            ),
            Particle(
                mass=1e-27,
                charge=-1.6e-19,
                position=np.array([5.0, 0.0]),
                velocity=np.array([-1000.0, 0.0]),
                color='blue'
            )
        ]
        
        # Create magnetic field
        magnetic_field = MagneticField.create_uniform_field(1.0)
        
        # Create simulation
        simulation = Simulation(
            particles=particles,
            magnetic_field=magnetic_field,
            timestep=0.01,
            duration=1.0,  # Short duration for testing
            collisions_enabled=True,
            decay_enabled=True
        )
        
        print(f"Simulation created with {len(particles)} particles")
        print(f"  Duration: {simulation.duration} s")
        print(f"  Time step: {simulation.timestep} s")
        
        # Run simulation
        results = simulation.run()
        print(f"Simulation completed")
        print(f"  Final time: {results['final_time']:.2f} s")
        print(f"  Active particles: {results['active_particles']}")
        print(f"  Total collisions: {results['total_collisions']}")
        print(f"  Total decays: {results['total_decays']}")
        
        return True
        
    except Exception as e:
        print(f"Simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualizer():
    """Test visualizer creation."""
    print("\nTesting visualizer...")
    
    try:
        from particle import Particle
        from fields import MagneticField
        from simulation import Simulation
        from visualizer import Visualizer
        
        # Create a simple simulation
        particles = [
            Particle(
                mass=1e-27,
                charge=1.6e-19,
                position=np.array([0.0, 0.0]),
                velocity=np.array([1000.0, 0.0]),
                color='red'
            )
        ]
        
        magnetic_field = MagneticField.create_uniform_field(1.0)
        
        simulation = Simulation(
            particles=particles,
            magnetic_field=magnetic_field,
            timestep=0.01,
            duration=0.5,  # Very short for testing
            collisions_enabled=False,
            decay_enabled=False
        )
        
        # Run simulation
        simulation.run()
        
        # Create visualizer
        visualizer = Visualizer(simulation)
        print("Visualizer created successfully")
        
        # Test trajectory data
        trajectories = simulation.get_particle_trajectories()
        print(f"Trajectory data extracted: {len(trajectories)} particles")
        
        return True
        
    except Exception as e:
        print(f"Visualizer test failed: {e}")
        return False

def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        # Test cyclotron frequency calculation
        freq = calculate_cyclotron_frequency(
            charge=1.6e-19,
            mass=1e-27,
            magnetic_field=1.0
        )
        print(f"Cyclotron frequency calculated: {freq:.2e} Hz")
        
        # Test Larmor radius calculation
        radius = calculate_larmor_radius(
            velocity=1000.0,
            charge=1.6e-19,
            mass=1e-27,
            magnetic_field=1.0
        )
        print(f"Larmor radius calculated: {radius:.2e} m")
        
        return True
        
    except Exception as e:
        print(f"Utility functions test failed: {e}")
        return False

def test_gui_error_fix():
    """Test that the GUI error is fixed."""
    print("\nTesting GUI error fix...")
    
    try:
        from particles import QuantumParticle
        from simulation import Simulation
        from fields import MagneticField
        from visualizer import Visualizer
        
        # Create a simple simulation
        particles = [
            QuantumParticle(
                particle_type='electron',
                position=np.array([0.0, 0.0]),
                velocity=np.array([1.0, 0.0]),
                lifetime=5.0
            )
        ]
        
        # Create magnetic field
        magnetic_field = MagneticField.create_uniform_field(1.0)
        
        # Create simulation
        simulation = Simulation(
            particles=list(particles),  # Cast to list of base Particle type
            magnetic_field=magnetic_field,
            duration=2.0,
            hep_analysis_enabled=True
        )
        
        print("  Running simulation...")
        results = simulation.run()
        
        print("  Creating visualizer...")
        viz = Visualizer(simulation)
        
        print("  Testing animation (close window to test error handling)...")
        # This should not cause an error when window is closed
        viz.animate(interval=100)
        print("  Animation completed successfully!")
        
        print("  Testing plot generation...")
        viz.generate_plots("test_output")
        viz.create_summary_plot("test_output")
        print("  Plot generation completed successfully!")
        
        print("GUI error fix test passed")
        return True
        
    except Exception as e:
        print(f"GUI error fix test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running tests for Particle Simulator")
    print()
    
    all_passed = test_imports()
    all_passed = all_passed and test_particle_creation()
    all_passed = all_passed and test_magnetic_field()
    all_passed = all_passed and test_simulation()
    all_passed = all_passed and test_visualizer()
    all_passed = all_passed and test_utils()
    all_passed = all_passed and test_gui_error_fix()
    
    print()
    print("=== Test Results ===")
    
    if all_passed:
        print("All tests passed! The application is ready to use.")
        print()
        print("To run the application:")
        print("  python main.py")
        print()
        print("Or run specific demos:")
        print("  python test_simulations/collider.py")
        print("  python test_simulations/plasma.py")
    else:
        print("Some tests failed. Please check the error messages above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 