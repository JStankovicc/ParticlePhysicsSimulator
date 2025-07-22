#!/usr/bin/env python3
"""
Quantum Particle Simulator in Magnetic Field
Main application for quantum particle simulation.
"""

import sys
import os

def main():
    print("=== Quantum Particle Simulator ===")
    print("Advanced particle physics simulation with HEP analysis")
    print()
    print("Choose what you'd like to do:")
    print("1. Open interactive GUI (recommended for beginners)")
    print("2. Run demo simulations (see different particle behaviors)")
    print("3. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == '1':
                print()
                print("Starting interactive GUI - you can create custom simulations...")
                try:
                    from gui import QuantumParticleGUI
                    app = QuantumParticleGUI()
                    app.run()
                except ImportError as e:
                    print(f"GUI not available: {e}")
                    print("To install GUI support: sudo pacman -S tk")
                break
                
            elif choice == '2':
                print()
                print("Available demo simulations:")
                print("1. Particle Collider - High-energy particle collisions")
                print("2. Plasma Simulation - Dense particle interactions")
                print("3. Tornado Simulation - Spiral particle motion")
                print("4. Explosion Simulation - Particle explosion effects")
                print("5. Run all demos")
                print("6. Back to main menu")
                
                demo_choice = input("Enter demo choice (1-6): ").strip()
                
                if demo_choice == '1':
                    print("Starting particle collider simulation...")
                    os.system("python test_simulations/collider.py")
                elif demo_choice == '2':
                    print("Starting plasma simulation...")
                    os.system("python test_simulations/plasma.py")
                elif demo_choice == '3':
                    print("Starting tornado simulation...")
                    os.system("python test_simulations/tornado.py")
                elif demo_choice == '4':
                    print("Starting explosion simulation...")
                    os.system("python test_simulations/explosion.py")
                elif demo_choice == '5':
                    print("Running all demo simulations...")
                    os.system("python test_simulations/run_demos.py")
                elif demo_choice == '6':
                    continue
                else:
                    print("Invalid choice! Please enter 1-6.")
                    
            elif choice == '3':
                print("Thanks for using the Quantum Particle Simulator!")
                break
                
            else:
                print("Invalid choice! Please enter 1-3.")
                
        except KeyboardInterrupt:
            print()
            print("Thanks for using the Quantum Particle Simulator!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 