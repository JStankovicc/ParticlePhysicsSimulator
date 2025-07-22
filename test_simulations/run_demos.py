#!/usr/bin/env python3
"""
Run Quantum Demo Simulations
Executes impressive quantum particle demo simulations.
"""

import sys
import os
import subprocess
import time

def run_demo(demo_name, script_path):
    """Run a quantum demo simulation."""
    print(f"\n{'='*60}")
    print(f"Starting: {demo_name}")
    print(f"{'='*60}")
    
    try:
        # Run the demo script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"\n{demo_name} completed successfully!")
        else:
            print(f"\n{demo_name} encountered an error (exit code {result.returncode})")
            
    except Exception as e:
        print(f"\nError running {demo_name}: {e}")
    
    # Wait a bit between demos
    time.sleep(2)

def main():
    print("=== Quantum Particle Demo Suite ===")
    print("Running a series of impressive particle physics simulations")
    print("Each demo shows different aspects of quantum particle behavior")
    print()
    
    # Define quantum demos
    demos = [
        ("Quantum Particle Collider", "collider.py"),
        ("Quantum Plasma Simulation", "plasma.py"),
        ("Quantum Tornado Simulation", "tornado.py"),
        ("Quantum Explosion Simulation", "explosion.py")
    ]
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Demo simulations to run:")
    for i, (name, script) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print()
    
    # Run all demos
    for name, script in demos:
        script_path = os.path.join(script_dir, script)
        if os.path.exists(script_path):
            run_demo(name, script_path)
        else:
            print(f"\nScript not found: {script_path}")
    
    print(f"\n{'='*60}")
    print("All quantum particle demos have finished!")
    print()
    print("Analysis results and visualizations have been saved to:")
    print("  output_graphs/ directory")
    print()
    print("Each simulation created comprehensive plots including:")
    print("  - Particle trajectory animations")
    print("  - Energy and collision analysis")
    print("  - HEP detector simulations")
    print("  - Statistical analysis reports")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 