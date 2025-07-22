import numpy as np
from typing import Dict, Optional
from particle import Particle

class QuantumParticle(Particle):
    """
    Extends the base Particle class to include quantum properties like spin and type,
    using a predefined dictionary of known particle types.
    """
    
    # A dictionary of predefined particle types with their physical properties.
    PARTICLE_TYPES = {
        'electron': {
            'name': 'Electron',
            'mass': 9.1093837015e-31,  # kg
            'charge': -1.602176634e-19,  # C
            'spin': 0.5,
            'color': 'cyan',
            'description': 'Fundamental fermion, negative charge'
        },
        'proton': {
            'name': 'Proton',
            'mass': 1.67262192369e-27,  # kg
            'charge': 1.602176634e-19,  # C
            'spin': 0.5,
            'color': 'magenta',
            'description': 'Composite fermion, positive charge'
        },
        'neutron': {
            'name': 'Neutron',
            'mass': 1.67492749804e-27,  # kg
            'charge': 0.0,  # C
            'spin': 0.5,
            'color': 'lime',
            'description': 'Composite fermion, neutral charge'
        },
        'photon': {
            'name': 'Photon',
            'mass': 0.0,  # kg (massless)
            'charge': 0.0,  # C
            'spin': 1.0,
            'color': 'gold',
            'description': 'Gauge boson, massless'
        },
        'quark_up': {
            'name': 'Up Quark',
            'mass': 2.2e-30,  # kg (approximate)
            'charge': 2/3 * 1.602176634e-19,  # C
            'spin': 0.5,
            'color': 'hotpink',
            'description': 'Elementary fermion, fractional charge'
        },
        'quark_down': {
            'name': 'Down Quark',
            'mass': 4.7e-30,  # kg (approximate)
            'charge': -1/3 * 1.602176634e-19,  # C
            'spin': 0.5,
            'color': 'coral',
            'description': 'Elementary fermion, fractional charge'
        },
        'custom': {
            'name': 'Custom Particle',
            'mass': 1e-30,  # kg (default)
            'charge': 0.0,  # C (default)
            'spin': 0.5,  # default
            'color': 'springgreen',
            'description': 'User-defined particle properties'
        }
    }
    
    def __init__(self, particle_type: str, position: np.ndarray, velocity: np.ndarray, 
                 lifetime: float = 10.0, custom_properties: Optional[Dict] = None):
        """
        Initializes a QuantumParticle instance.
        
        Args:
            particle_type: Type of quantum particle ('electron', 'proton', etc., or 'custom')
            position: Initial position [x, y] in μm
            velocity: Initial velocity [vx, vy] in μm/s
            lifetime: Particle lifetime in seconds
            custom_properties: Optional custom properties to override defaults
        """
        if particle_type not in self.PARTICLE_TYPES:
            raise ValueError(f"Unknown particle type: {particle_type}")
            
        particle_data = self.PARTICLE_TYPES[particle_type].copy()
        
        # Allow overriding predefined properties with custom values.
        if custom_properties:
            particle_data.update(custom_properties)
        
        # Convert standard physical units (kg, C) to the simulation's internal scale (μm-based).
        mass_kg = particle_data['mass']
        charge_c = particle_data['charge']
        
        mass_scaled = mass_kg * 1e12  # Scale mass for μm simulation
        charge_scaled = charge_c * 1e6  # Scale charge for μm simulation
        
        super().__init__(
            mass=mass_scaled,
            charge=charge_scaled,
            position=position,
            velocity=velocity,
            lifetime=lifetime,
            color=particle_data['color'],
            radius=0.01  # Standard radius for visualization and collision.
        )
        
        # Store the specific quantum properties for this particle instance.
        self.particle_type = particle_type
        self.quantum_properties = particle_data
        self.spin = particle_data['spin']
        self.description = particle_data['description']
        
    def get_quantum_info(self) -> Dict:
        """Returns a dictionary with the particle's quantum properties."""
        return {
            'type': self.particle_type,
            'name': self.quantum_properties['name'],
            'mass_kg': self.quantum_properties['mass'],
            'charge_c': self.quantum_properties['charge'],
            'spin': self.spin,
            'color': self.color,
            'description': self.description,
            'position_um': self.position.tolist(),
            'velocity_um_s': self.velocity.tolist(),
            'lifetime_s': self.lifetime
        }
        
    def __str__(self) -> str:
        """Provides a string representation of the quantum particle."""
        return f"QuantumParticle({self.particle_type}, pos={self.position}, vel={self.velocity})"
        
    @classmethod
    def get_available_types(cls) -> Dict[str, Dict]:
        """Returns a copy of the dictionary of predefined particle types."""
        return cls.PARTICLE_TYPES.copy()
        
    @classmethod
    def create_particle(cls, particle_type: str, position: np.ndarray, velocity: np.ndarray,
                       lifetime: float = 10.0, **custom_properties) -> 'QuantumParticle':
        """
        A factory method to create a quantum particle, simplifying instance creation.
        
        Args:
            particle_type: Type of particle
            position: Initial position [x, y] in μm
            velocity: Initial velocity [vx, vy] in μm/s
            lifetime: Particle lifetime in seconds
            **custom_properties: Custom properties to override defaults
            
        Returns:
            QuantumParticle instance
        """
        return cls(particle_type, position, velocity, lifetime, custom_properties) 