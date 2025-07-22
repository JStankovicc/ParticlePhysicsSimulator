import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import uuid

class Particle:
    """
    Represents a classical particle in 2D space, subject to forces and collisions.
    """
    
    def __init__(self, mass: float, charge: float, position: np.ndarray, 
                 velocity: np.ndarray, lifetime: Optional[float] = None,
                 collidable: bool = True, color: str = 'blue', radius: float = 0.1):
        """
        Initializes a Particle instance.
        
        Args:
            mass: Particle mass in kg
            charge: Particle charge in C
            position: Initial position [x, y] in m
            velocity: Initial velocity [vx, vy] in m/s
            lifetime: Time before decay in seconds (None for no decay)
            collidable: Whether particle can collide with others
            color: Color for visualization
            radius: Radius for collision detection and visualization
        """
        self.id = str(uuid.uuid4())
        self.mass = mass
        self.charge = charge
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.array([0.0, 0.0])
        self.lifetime = lifetime
        self.decayed = False
        self.collidable = collidable
        self.color = color
        self.radius = radius
        self.history = {
            'time': [],
            'position': [],
            'velocity': [],
            'acceleration': [],
            'energy': []
        }
        self.creation_time = 0.0
        
    def update(self, dt: float, magnetic_field: np.ndarray, current_time: float) -> None:
        """
        Updates the particle's position and velocity using Lorentz force.
        
        Args:
            dt: Time step in seconds
            magnetic_field: Magnetic field vector [Bx, By, Bz] in Tesla
            current_time: Current simulation time
        """
        # Do not update decayed particles.
        if self.decayed:
            return
            
        # First, check if the particle decays at the current time.
        if self.check_decay(current_time):
            return
             
        # The Lorentz force is calculated based on the particle's charge, velocity,
        # and the magnetic field (assumed to be perpendicular to the 2D plane).
        Bz = magnetic_field[2] if len(magnetic_field) > 2 else magnetic_field[0]
        
        # Cross product in 2D: v × B = [vx, vy] × [0, 0, Bz] = [vy*Bz, -vx*Bz]
        lorentz_force = self.charge * np.array([self.velocity[1] * Bz, -self.velocity[0] * Bz])
        
        # Acceleration is derived from the force (a = F/m), but massless particles
        # are not accelerated by the magnetic field.
        if self.mass > 0:
            self.acceleration = lorentz_force / self.mass
        else:
            # For massless particles, no acceleration from magnetic field
            self.acceleration = np.array([0.0, 0.0])
        
        # To prevent numerical instability, clip acceleration, velocity, and position
        # to reasonable maximum values.
        max_acceleration = 1e6  # Maximum reasonable acceleration in m/s²
        if np.any(np.abs(self.acceleration) > max_acceleration):
            self.acceleration = np.clip(self.acceleration, -max_acceleration, max_acceleration)
        
        velocity_change = self.acceleration * dt
        self.velocity += velocity_change
        
        # Check for overflow and limit velocity if necessary
        max_velocity = 1e4  # Maximum reasonable velocity in m/s
        if np.any(np.abs(self.velocity) > max_velocity):
            self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
        
        # Update position
        position_change = self.velocity * dt
        self.position += position_change
        
        # Check for overflow and limit position if necessary
        max_position = 1e6  # Maximum reasonable position in m
        if np.any(np.abs(self.position) > max_position):
            self.position = np.clip(self.position, -max_position, max_position)
        
        # Record the particle's state in its history log if all values are finite.
        if np.all(np.isfinite(self.position)) and np.all(np.isfinite(self.velocity)):
            self.history['time'].append(current_time)
            self.history['position'].append(self.position.copy())
            self.history['velocity'].append(self.velocity.copy())
            self.history['acceleration'].append(self.acceleration.copy())
            self.history['energy'].append(self.get_kinetic_energy())
        
    def check_decay(self, current_time: float) -> bool:
        """Checks if the particle should decay based on its lifetime.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            True if particle decayed, False otherwise
        """
        if self.lifetime is not None and current_time - self.creation_time >= self.lifetime:
            self.decayed = True
            return True
        return False
        
    def apply_force(self, force: np.ndarray) -> None:
        """Applies an external force to the particle.
        
        Args:
            force: Force vector [Fx, Fy] in N
        """
        if not self.decayed:
            self.acceleration += force / self.mass
            
    def distance_to(self, other: 'Particle') -> float:
        """Calculates the Euclidean distance to another particle.
        
        Args:
            other: Another particle
            
        Returns:
            Distance between particles in meters
        """
        return float(np.linalg.norm(self.position - other.position))
        
    def get_kinetic_energy(self) -> float:
        """Calculates the kinetic energy of the particle.
        
        Returns:
            Kinetic energy in Joules
        """
        return float(0.5 * self.mass * np.linalg.norm(self.velocity) ** 2)
        
    def get_momentum(self) -> np.ndarray:
        """Calculates the momentum of the particle.
        
        Returns:
            Momentum vector [px, py] in kg⋅m/s
        """
        return self.mass * self.velocity
        
    def is_colliding_with(self, other: 'Particle') -> bool:
        """Checks if this particle is colliding with another particle based on their radii.
        
        Args:
            other: Another particle
            
        Returns:
            True if particles are colliding
        """
        if not self.collidable or not other.collidable:
            return False
            
        distance = self.distance_to(other)
        return distance <= (self.radius + other.radius)
        
    def get_state(self) -> dict:
        """Returns the current state of the particle as a dictionary.
        
        Returns:
            Dictionary with particle state
        """
        return {
            'id': self.id,
            'mass': self.mass,
            'charge': self.charge,
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'acceleration': self.acceleration.copy(),
            'decayed': self.decayed,
            'energy': self.get_kinetic_energy()
        }
        
    def __str__(self) -> str:
        """String representation of the particle."""
        return f"Particle(id={self.id[:8]}, mass={self.mass:.2e}, charge={self.charge:.2e}, " \
               f"pos={self.position}, vel={self.velocity}, decayed={self.decayed})" 

    @property
    def is_active(self) -> bool:
        """Checks if the particle is currently active (i.e., not decayed).
        
        Returns:
            True if particle is active, False if decayed
        """
        return not self.decayed 