import numpy as np
from typing import Callable, Union, Tuple, Optional
import math

class MagneticField:
    """
    Defines a magnetic field, which can be uniform or spatially varying.
    """
    
    def __init__(self, field_type: str = 'uniform', strength: Union[float, Callable] = 1.0,
                 direction: Optional[np.ndarray] = None):
        """
        Initializes the magnetic field.
        
        Args:
            field_type: 'uniform' or 'non-uniform'
            strength: Field strength in Tesla (float for uniform, function for non-uniform)
            direction: Field direction vector [Bx, By, Bz] (default: [0, 0, 1])
        """
        self.type = field_type
        self.strength = strength
        
        if direction is None:
            # Default to a field perpendicular to the 2D plane (in the z-direction).
            self.direction = np.array([0.0, 0.0, 1.0])
        else:
            self.direction = np.array(direction)
            
        # Normalize the direction vector to ensure it represents only direction.
        if np.linalg.norm(self.direction) > 0:
            self.direction = self.direction / np.linalg.norm(self.direction)
            
    def get_field_at(self, position: np.ndarray) -> np.ndarray:
        """Calculates the magnetic field vector at a given position.
        
        Args:
            position: Position [x, y] in meters
            
        Returns:
            Magnetic field vector [Bx, By, Bz] in Tesla
        """
        if self.type == 'uniform':
            # For a uniform field, the vector is constant everywhere.
            magnitude = float(self.strength)
            return magnitude * self.direction
            
        elif self.type == 'non-uniform':
            # For a non-uniform field, the magnitude depends on the position.
            if callable(self.strength):
                magnitude = self.strength(position[0], position[1])
            else:
                # Default non-uniform field: B = B0 * sin(x) * cos(y)
                x, y = position
                magnitude = self.strength * math.sin(x) * math.cos(y)
                
            return magnitude * self.direction
            
        else:
            raise ValueError(f"Unknown field type: {self.type}")
            
    def get_field_magnitude_at(self, position: np.ndarray) -> float:
        """Calculates the magnitude of the magnetic field at a given position.
        
        Args:
            position: Position [x, y] in meters
            
        Returns:
            Field magnitude in Tesla
        """
        field_vector = self.get_field_at(position)
        return float(np.linalg.norm(field_vector))
        
    @staticmethod
    def create_uniform_field(strength: float = 1.0) -> 'MagneticField':
        """A factory method to create a uniform magnetic field.
        
        Args:
            strength: Field strength in Tesla
            
        Returns:
            A MagneticField instance with uniform properties.
        """
        return MagneticField('uniform', strength)
        
    @staticmethod
    def create_nonuniform_field(strength_function: Optional[Callable] = None, 
                               base_strength: float = 1.0) -> 'MagneticField':
        """A factory method to create a non-uniform magnetic field.
        
        Args:
            strength_function: Function B(x, y) that returns field strength
            base_strength: Base field strength for default function
            
        Returns:
            A MagneticField instance with non-uniform properties.
        """
        if strength_function is None:
            # Default non-uniform field: B = B0 * sin(x) * cos(y)
            def default_function(x, y):
                return base_strength * math.sin(x) * math.cos(y)
            strength_function = default_function
            
        return MagneticField('non-uniform', strength_function)
        
    @staticmethod
    def create_dipole_field(strength: float = 1.0, dipole_position: Optional[np.ndarray] = None) -> 'MagneticField':
        """A factory method to create a dipole magnetic field.
        
        Args:
            strength: Field strength in Tesla
            dipole_position: Position of dipole [x, y] (default: origin)
            
        Returns:
            A MagneticField instance configured as a dipole field.
        """
        if dipole_position is None:
            dipole_position = np.array([0.0, 0.0])
            
        def dipole_function(x, y):
            # Distance from dipole
            r = np.linalg.norm([x - dipole_position[0], y - dipole_position[1]])
            if r < 1e-10:  # Avoid division by zero
                return 0.0
            # Dipole field: B ∝ 1/r³
            return strength / (r ** 3)
            
        return MagneticField('non-uniform', dipole_function)
        
    @staticmethod
    def create_quadrupole_field(strength: float = 1.0) -> 'MagneticField':
        """A factory method to create a quadrupole magnetic field.
        
        Args:
            strength: Field strength in Tesla
            
        Returns:
            A MagneticField instance configured as a quadrupole field.
        """
        def quadrupole_function(x, y):
            # Quadrupole field: B ∝ x*y
            return strength * x * y
            
        return MagneticField('non-uniform', quadrupole_function)
        
    def get_info(self) -> dict:
        """Returns a dictionary with information about the magnetic field's properties.
        
        Returns:
            Dictionary with field information
        """
        return {
            'type': self.type,
            'strength': str(self.strength) if callable(self.strength) else self.strength,
            'direction': self.direction.tolist()
        }
        
    def __str__(self) -> str:
        """Provides a string representation of the magnetic field."""
        info = self.get_info()
        return f"MagneticField(type={info['type']}, strength={info['strength']}, direction={info['direction']})" 