#!/usr/bin/env python3
"""
Simulates a multi-layer High Energy Physics detector system for particle analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from particle import Particle
from particles import QuantumParticle

@dataclass
class DetectorHit:
    """Represents a detector hit from a particle."""
    particle_id: str
    particle_type: str
    energy: float
    position: np.ndarray
    time: float
    detector_layer: str
    efficiency_factor: float
    resolution_smearing: float

class HEPDetector:
    """
    Simulates a multi-layer HEP detector system similar to ATLAS/CMS.
    """
    
    def __init__(self):
        """Initialize the HEP detector system."""
        # Define detector layers from inner to outer components
        self.layers = {
            'inner_tracker': {
                'radius_range': (0.1, 500.0),  # μm - expanded for better coverage
                'efficiency': 0.95,
                'resolution': 0.02,  # 2% resolution
                'material': 'silicon',
                'purpose': 'track reconstruction'
            },
            'electromagnetic_calorimeter': {
                'radius_range': (500.0, 5000.0),  # μm - expanded coverage
                'efficiency': 0.90,
                'resolution': 0.05,  # 5% resolution
                'material': 'lead_tungstate',
                'purpose': 'electron/photon energy measurement'
            },
            'hadronic_calorimeter': {
                'radius_range': (5000.0, 15000.0),  # μm - expanded coverage
                'efficiency': 0.85,
                'resolution': 0.10,  # 10% resolution
                'material': 'iron_scintillator',
                'purpose': 'hadron energy measurement'
            },
            'muon_chambers': {
                'radius_range': (15000.0, 50000.0),  # μm - much larger coverage
                'efficiency': 0.80,
                'resolution': 0.08,  # 8% resolution
                'material': 'gas_chambers',
                'purpose': 'muon identification'
            },
            'outer_tracker': {
                'radius_range': (50000.0, 100000.0),  # μm - additional outer layer
                'efficiency': 0.75,
                'resolution': 0.12,  # 12% resolution
                'material': 'scintillator',
                'purpose': 'high-energy particle detection'
            }
        }
        
        # Detector response parameters
        self.noise_level = 0.01
        self.dead_time = 0.001   # 1ms dead time
        self.hits = []
        self.reconstructed_tracks = []
        
    def detect_particle(self, particle: Particle, time: float) -> List[DetectorHit]:
        """
        Simulate detector response to a particle.
        
        Args:
            particle: Particle to detect
            time: Time of detection
            
        Returns:
            List of detector hits
        """
        hits = []
        particle_radius = np.linalg.norm(particle.position)
        
        # Check which detector layers the particle traverses
        for layer_name, layer_config in self.layers.items():
            r_min, r_max = layer_config['radius_range']
            
            # Check if particle is within layer radius range
            if r_min <= particle_radius <= r_max:
                # Higher detection probability for active particles
                detection_prob = layer_config['efficiency']
                if not particle.decayed: # Active particles get detection boost
                    detection_prob *= 1.2  # 20% boost for active particles
                
                # Check if particle is detected (efficiency)
                if random.random() < min(detection_prob, 1.0):
                    # Calculate detected energy with resolution smearing
                    true_energy = particle.get_kinetic_energy()
                    if true_energy > 0:  # Only detect particles with energy
                        resolution = layer_config['resolution']
                        smeared_energy = true_energy * (1 + np.random.normal(0, resolution))
                        
                        # Add electronic noise
                        noise = np.random.normal(0, self.noise_level * true_energy)
                        detected_energy = max(0, smeared_energy + noise)
                        
                        # Record detector hit
                        hit = DetectorHit(
                            particle_id=particle.id,
                            particle_type=self._identify_particle_type(particle),
                            energy=detected_energy,
                            position=particle.position.copy(),
                            time=time,
                            detector_layer=layer_name,
                            efficiency_factor=layer_config['efficiency'],
                            resolution_smearing=resolution
                        )
                        hits.append(hit)
        
        # Simulate random background noise hits
        if random.random() < 0.01:  # 1% chance of background hit
            background_layer = random.choice(list(self.layers.keys()))
            layer_config = self.layers[background_layer]
            r_min, r_max = layer_config['radius_range']
            
            # Background hit at random position within layer
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(r_min, r_max)
            bg_position = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            
            bg_hit = DetectorHit(
                particle_id="background",
                particle_type="background",
                energy=random.uniform(0.1, 1.0) * 1e-18,  # Small background energy
                position=bg_position,
                time=time,
                detector_layer=background_layer,
                efficiency_factor=0.1,
                resolution_smearing=0.2
            )
            hits.append(bg_hit)
        
        self.hits.extend(hits)
        return hits
    
    def _identify_particle_type(self, particle: Particle) -> str:
        """Identify particle type for hit labeling."""
        # Use particle_type attribute if available (QuantumParticle)
        particle_type = getattr(particle, 'particle_type', None)
        if particle_type:
            return str(particle_type)
        elif particle.charge == 0:
            if particle.mass == 0:
                return 'photon'
            else:
                return 'neutron'
        elif particle.charge > 0:
            return 'positive_charged'
        else:
            return 'negative_charged'
    
    def reconstruct_tracks(self) -> List[Dict]:
        """
        Reconstruct particle tracks from detector hits.
        
        Returns:
            List of reconstructed track dictionaries
        """
        tracks = []
        
        # Group hits by particle ID
        particle_hits = {}
        for hit in self.hits:
            if hit.particle_id not in particle_hits:
                particle_hits[hit.particle_id] = []
            particle_hits[hit.particle_id].append(hit)
         
        # Create tracks for particles with sufficient hits
        for particle_id, hits in particle_hits.items():
            if len(hits) >= 2:  # Need at least 2 hits for track
                # Sort hits by detector layer order (inner to outer)
                layer_order = ['inner_tracker', 'electromagnetic_calorimeter', 
                               'hadronic_calorimeter', 'muon_chambers', 'outer_tracker']
                # Filter and sort valid hits
                valid_hits = [h for h in hits if h.detector_layer in layer_order]
                valid_hits.sort(key=lambda h: layer_order.index(h.detector_layer))
                hits = valid_hits
                
                # Calculate track parameters
                total_energy = sum(hit.energy for hit in hits)
                momentum = self._calculate_momentum(hits)
                track_length = self._calculate_track_length(hits)
                
                track = {
                    'particle_id': particle_id,
                    'particle_type': hits[0].particle_type,
                    'total_energy': total_energy,
                    'momentum': momentum,
                    'track_length': track_length,
                    'hits': hits,
                    'detector_layers': [hit.detector_layer for hit in hits],
                    'reconstruction_quality': self._assess_track_quality(hits)
                }
                tracks.append(track)
        
        self.reconstructed_tracks = tracks
        return tracks
    
    def _calculate_momentum(self, hits: List[DetectorHit]) -> float:
        """Estimate particle momentum from detector hits."""
        if len(hits) < 2:
            return 0.0
        
        # Simplified momentum estimation
        # Real calculation would analyze track curvature in magnetic field
        total_energy = sum(hit.energy for hit in hits)
        return total_energy * 0.8  # Rough approximation
    
    def _calculate_track_length(self, hits: List[DetectorHit]) -> float:
        """Calculate total track length from hits."""
        if len(hits) < 2:
            return 0.0
        
        length = 0.0
        for i in range(1, len(hits)):
            distance = np.linalg.norm(hits[i].position - hits[i-1].position)
            length += float(distance)
        return float(length)
    
    def _assess_track_quality(self, hits: List[DetectorHit]) -> float:
        """Assess track reconstruction quality."""
        if not hits:
            return 0.0
        
        # Quality based on hits, layers, efficiency, and resolution
        quality = len(hits) / len(self.layers)
        
        avg_efficiency = np.mean([hit.efficiency_factor for hit in hits])
        quality *= float(avg_efficiency)
        
        avg_resolution = np.mean([hit.resolution_smearing for hit in hits])
        quality *= (1 - float(avg_resolution))
        
        return float(min(1.0, quality))
    
    def get_detector_summary(self) -> Dict:
        """Return detector performance summary."""
        return {
            'total_hits': len(self.hits),
            'reconstructed_tracks': len(self.reconstructed_tracks),
            'average_track_quality': np.mean([t['reconstruction_quality'] for t in self.reconstructed_tracks]) if self.reconstructed_tracks else 0.0,
            'detector_efficiency': len(self.reconstructed_tracks) / max(1, len(set(hit.particle_id for hit in self.hits))),
            'energy_resolution': np.std([hit.energy for hit in self.hits]) / np.mean([hit.energy for hit in self.hits]) if self.hits else 0.0
        }
    
    def clear_data(self):
        """Clear all stored hits and tracks for new simulation."""
        self.hits.clear()
        self.reconstructed_tracks.clear() 