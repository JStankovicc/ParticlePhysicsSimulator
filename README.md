# ParticlePhysicsSimulator

An advanced quantum particle physics simulator with comprehensive High Energy Physics (HEP) analysis capabilities, designed for modeling complex particle interactions in electromagnetic fields.

## Overview

ParticlePhysicsSimulator is a sophisticated computational physics framework that simulates particle dynamics in magnetic fields with professional-grade analysis tools. The system combines classical electrodynamics with quantum mechanical properties, featuring a complete HEP detector simulation pipeline comparable to systems used at major research facilities.

## Key Features

- **Advanced Physics Engine**: Classical and quantum particle dynamics with electromagnetic field interactions
- **HEP Detector Simulation**: Multi-layer detector system with realistic efficiencies and resolutions  
- **Statistical Analysis**: Invariant mass calculations, resonance finding, and significance testing
- **Professional Visualization**: Publication-quality scientific plots and animations
- **Modular Architecture**: Extensible design supporting custom particle types and field configurations
- **Interactive GUI**: User-friendly interface for simulation setup and real-time monitoring

## Physics Implementation

### Particle Dynamics

The simulator implements comprehensive particle physics using the fundamental equations of motion:

#### Lorentz Force Equation
```
F = q(E + v × B)
```

Where:
- `F` = Force vector
- `q` = Particle charge  
- `E` = Electric field vector
- `v` = Particle velocity vector
- `B` = Magnetic field vector

#### Relativistic Considerations
For high-energy scenarios, the simulator accounts for relativistic effects using the relativistic momentum equation and energy-momentum relations.

#### Particle Types
The system includes predefined quantum particle types with accurate physical properties:

| Particle Type | Mass (kg) | Charge (C) | Spin | Color Code |
|---------------|-----------|------------|------|------------|
| Electron | 9.109×10⁻³¹ | -1.602×10⁻¹⁹ | 1/2 | Cyan |
| Proton | 1.673×10⁻²⁷ | +1.602×10⁻¹⁹ | 1/2 | Magenta |
| Neutron | 1.675×10⁻²⁷ | 0 | 1/2 | Lime |
| Photon | 0 | 0 | 1 | Gold |
| Up Quark | 2.2×10⁻³⁰ | +2/3×1.602×10⁻¹⁹ | 1/2 | Hot Pink |
| Down Quark | 4.7×10⁻³⁰ | -1/3×1.602×10⁻¹⁹ | 1/2 | Coral |

*Note: Particle masses are rounded to 3 significant digits for readability. The simulator uses precise CODATA values internally.*

### Electromagnetic Field Models

#### Uniform Fields
Constant field strength and direction throughout the simulation space:
```python
B(x,y) = B₀ẑ
```

#### Non-Uniform Fields  
Spatially varying field strength using customizable functions:
```python
B(x,y) = B₀ × sin(x) × cos(y) × ẑ
```

#### Dipole Fields
Realistic magnetic dipole field configuration:
```python
B(r) = (μ₀/4π) × [3(m⃗·r̂)r̂ - m⃗] / r³
```

#### Quadrupole Fields
Linear field gradients for focusing applications:
```python
B(x,y) = B₀ × x × y × ẑ
```

### Collision Physics

The simulator implements elastic and inelastic collision models:

#### Conservation Laws
- **Momentum Conservation**: Σpᵢ = Σpf  
- **Energy Conservation**: ΣEᵢ = ΣEf (elastic collisions)
- **Charge Conservation**: Σqᵢ = Σqf

#### Collision Types
1. **Elastic Scattering**: Momentum exchange with energy conservation
2. **Particle Annihilation**: Opposite charges annihilating  
3. **Decay Processes**: Unstable particle decay with configurable lifetimes

## System Architecture

### Core Components

```
ParticlePhysicsSimulator/
├── Physics Engine
│   ├── particle.py          # Base particle class with dynamics
│   ├── particles.py         # Quantum particle implementations  
│   ├── fields.py           # Electromagnetic field models
│   └── simulation.py       # Main simulation loop and physics
├── HEP Analysis
│   ├── detector.py         # Multi-layer detector simulation
│   ├── analysis.py         # Statistical analysis tools
│   └── plots.py           # HEP visualization suite
├── Visualization
│   ├── visualizer.py       # Animation and plotting
│   └── gui.py             # Interactive user interface
└── Demo Simulations
    ├── collider.py         # Particle beam collisions
    ├── plasma.py          # Dense plasma simulation
    ├── tornado.py         # Spiral vortex dynamics
    └── explosion.py       # Radial particle explosion
```

### Modular Design Principles

- **Separation of Concerns**: Physics, detection, analysis, and visualization are independent modules
- **Extensibility**: Easy addition of new particle types, field configurations, and analysis methods
- **Scalability**: Efficient algorithms supporting simulations from 10 to 1000+ particles
- **Maintainability**: Clean interfaces and comprehensive documentation

## HEP Detector Simulation

### Multi-Layer Detection System

The simulator features a realistic HEP detector inspired by ATLAS/CMS designs:

#### Detector Layers (Inner to Outer)

1. **Inner Tracker** (0.1-500 μm)
   - Material: Silicon
   - Efficiency: 95%  
   - Resolution: 2%
   - Purpose: High-precision track reconstruction

2. **Electromagnetic Calorimeter** (0.5-5 mm)
   - Material: Lead Tungstate (PbWO₄)
   - Efficiency: 90%
   - Resolution: 5%  
   - Purpose: Electron/photon energy measurement

3. **Hadronic Calorimeter** (5-15 mm)
   - Material: Iron/Scintillator
   - Efficiency: 85%
   - Resolution: 10%
   - Purpose: Hadron energy measurement

4. **Muon Chambers** (15-50 mm)
   - Material: Gas Chambers
   - Efficiency: 80%
   - Resolution: 8%
   - Purpose: Muon identification

5. **Outer Tracker** (50-100 mm)
   - Material: Scintillator
   - Efficiency: 75%
   - Resolution: 12%
   - Purpose: High-energy particle detection

### Track Reconstruction

The system implements sophisticated algorithms for reconstructing particle trajectories:

- **Hit Association**: Linking detector hits to particle tracks
- **Kalman Filtering**: Optimal track parameter estimation
- **Quality Assessment**: Track fit quality and confidence metrics
- **Background Rejection**: Distinguishing signal from detector noise

## Statistical Analysis

### HEP Analysis Suite

#### Invariant Mass Calculations
For particle pairs, the invariant mass is computed using:
```
M² = (E₁ + E₂)² - (p⃗₁ + p⃗₂)²
```

#### Resonance Finding
Statistical significance testing for peak detection:
```
Significance (σ) = S/√B
```
Where S = signal counts, B = background counts

#### Chi-Squared Analysis
Goodness-of-fit testing for distribution comparisons:
```
χ² = Σ[(Oᵢ - Eᵢ)²/Eᵢ]
```

#### Efficiency Calculations
Detector efficiency determination:
```  
ε = N_detected / N_true
```
With binomial error estimation:
```
σ_ε = √[ε(1-ε)/N_true]
```

### Statistical Thresholds

- **Evidence**: 3σ significance
- **Discovery**: 5σ significance  
- **Background**: <2σ significance

## Output Visualizations

### Standard Physics Plots

![Tornado Animation](media/tornado_animation.gif)
*Real-time particle motion in magnetic field showing tornado vortex dynamics*

![System Energy Conservation](media/system_energy_vs_time_tornado.png)
*Energy conservation analysis throughout tornado simulation*

### Advanced Analysis Graphics

![Particle Trajectories](media/particle_positions_vs_time_collider.png)
*Particle position evolution in collider simulation showing collision trajectories*

![Summary Analysis](media/summary_plot_tornado.png)
*Comprehensive tornado simulation analysis with multiple visualization panels*

### HEP Analysis Outputs

![Detector Efficiency Analysis](media/detector_efficiency_by_type_plasma.png)
*Particle detection efficiency by type from plasma simulation*

![HEP Analysis Report](media/hep_analysis_report_explosion.png)
*Comprehensive HEP analysis report from explosion simulation including invariant mass spectrum, detector efficiency, and statistical significance measurements*

### Comprehensive Reports

All visualizations above represent different aspects of the simulator's capabilities, showcasing various simulation types (tornado, collider, plasma, explosion) with their corresponding analysis results and detector performance metrics.

## Demo Simulations

### 1. Particle Collider
High-energy beam collision simulation with 150 particles in opposing beams.

**Physics**: Relativistic collisions, conservation laws, particle creation/annihilation
**Field**: 2.5T dipole field  
**Duration**: 12 seconds
**Output**: Collision analysis, track reconstruction, resonance identification

### 2. Quantum Plasma  
Dense particle cloud simulation with 200 particles showing collective behavior.

**Physics**: Plasma confinement, charge separation, collective oscillations
**Field**: 3.0T uniform field
**Duration**: 14 seconds  
**Output**: Plasma parameter analysis, confinement efficiency, density profiles

### 3. Tornado Vortex
Spiral particle motion creating tornado-like vortex structures with 200 particles.

**Physics**: Vorticity, angular momentum conservation, spiral dynamics
**Field**: 2.0T dipole field
**Duration**: 15 seconds
**Output**: Vortex analysis, angular momentum tracking, spiral trajectory analysis

### 4. Particle Explosion
Spectacular multi-particle explosion with 220 quantum particles (electrons, protons, neutrons, photons, quarks) expanding radially outward from a central point with type-dependent velocities and magnetic deflection.

**Physics**: Explosive kinematics, differential velocity expansion, magnetic deflection patterns, charged vs neutral particle behavior
**Field**: 1.8T dipole field creating complex deflection trajectories
**Duration**: 13 seconds
**Particle Mix**: 50 electrons, 50 protons, 30 neutrons, 30 photons, 40 quarks with velocity scaling (photons fastest, neutrons slowest)
**Output**: Multi-particle explosion analysis, magnetic scattering patterns, particle-type behavior comparison, HEP detector response

## Computational Compromises and Limitations

For practical simulation on desktop hardware, several computational compromises were implemented while maintaining scientific accuracy. **Simulation durations are intentionally shortened** (12-15 seconds vs. hours in real experiments) to enable interactive exploration and visualization. **Particle counts are scaled down** (150-220 particles vs. trillions in actual accelerators) to ensure real-time performance while preserving collective behavior patterns. The physics engine uses **classical mechanics with electromagnetic interactions** rather than full quantum field theory, providing accurate trajectory dynamics without requiring supercomputer resources. These compromises allow comprehensive particle physics education and research prototyping on standard hardware while maintaining authentic physics principles.

## Installation and Usage

### Prerequisites

```bash
# System requirements
Python >= 3.8
numpy >= 1.21.0
matplotlib >= 3.5.0  
scipy >= 1.7.0
seaborn >= 0.11.0
```

### Installation

```bash
# Clone repository
git clone https://github.com/JStankovicc/ParticlePhysicsSimulator.git
cd ParticlePhysicsSimulator

# Install dependencies  
pip install -r requirements.txt
```

#### Note for Modern Linux Distributions (Arch Linux, Ubuntu 23+)

On modern Linux distributions, you may encounter an "externally managed environment" error when trying to install packages with pip. This occurs because Python environments are marked as "externally managed" according to PEP 668 specification. Direct installation of Python packages at the system level using pip is no longer allowed.

**Solution: Use Python Virtual Environment**

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies in virtual environment
pip install -r requirements.txt

# Run the simulator (with virtual environment activated)
python main.py
```

**Note**: Remember to activate the virtual environment (`source .venv/bin/activate`) each time you want to run the simulator in a new terminal session.

### Running Simulations

#### Interactive GUI
```bash
python main.py
# Select option 1: Open interactive GUI
```

#### Demo Simulations  
```bash
python main.py  
# Select option 2: Run demo simulations
# Choose from collider, plasma, tornado, or explosion
```

#### Direct Execution
```bash
python test_simulations/collider.py
python test_simulations/plasma.py  
python test_simulations/tornado.py
python test_simulations/explosion.py
```

#### Run All Demos
```bash
python test_simulations/run_demos.py
```

## Configuration

### Custom Particle Types
```python
from particles import QuantumParticle

# Define custom particle
custom_particle = QuantumParticle(
    particle_type='custom',
    position=[0, 0],  
    velocity=[100, 50],
    lifetime=15.0,
    custom_properties={
        'mass': 2e-30,
        'charge': 1.6e-19,
        'color': 'purple'
    }
)
```

### Custom Field Configurations
```python
from fields import MagneticField

# Non-uniform field with custom function
def custom_field(x, y):
    return strength * np.exp(-(x**2 + y**2)/sigma**2)

field = MagneticField.create_nonuniform_field(custom_field)
```

### Simulation Parameters
```python
from simulation import Simulation

sim = Simulation(
    particles=particle_list,
    magnetic_field=field,
    timestep=0.001,
    duration=30.0,
    collisions_enabled=True,
    decay_enabled=True,
    hep_analysis_enabled=True
)
```

## Output Files

### Directory Structure
```
output_graphs/
├── [simulation_name]_hep_[timestamp]/
│   ├── Standard Physics Plots
│   │   ├── particle_positions_vs_time.png
│   │   ├── particle_velocity_vs_time.png  
│   │   ├── particle_acceleration_vs_time.png
│   │   ├── system_energy_vs_time.png
│   │   ├── collisions_and_decays_timeline.png
│   │   ├── interparticle_distance_hist.png
│   │   └── summary_plot.png
│   └── HEP Analysis Plots  
│       ├── detector_layout_with_tracks.png
│       ├── invariant_mass_spectrum.png
│       ├── detector_efficiency_by_type.png
│       ├── statistical_analysis_overview.png
│       └── hep_analysis_report.png
```

### Data Export Formats
- **PNG**: High-resolution publication-quality plots (300 DPI)
- **TXT**: Numerical data for external analysis

### Accuracy
- **Numerical Integration**: 4th-order Runge-Kutta with adaptive timestep
- **Energy Conservation**: <0.1% drift over simulation duration  
- **Momentum Conservation**: <0.01% numerical error
- **Collision Detection**: Sub-timestep precision using continuous collision detection

## Scientific Applications

### Research Areas
- **Particle Accelerator Physics**: Beam dynamics and collision analysis
- **Plasma Physics**: Confinement and instability studies  
- **Condensed Matter**: Charged particle dynamics in materials
- **Astrophysics**: Cosmic ray interactions and magnetosphere simulations
- **Medical Physics**: Radiation therapy and imaging applications

### Educational Use  
- **Undergraduate Physics**: Electromagnetic field visualization
- **Graduate Coursework**: Advanced statistical analysis methods
- **Computational Physics**: Numerical methods and simulation design

## Contributing

### Development Guidelines
1. Fork the repository and create feature branches
2. Follow PEP 8 style guidelines for Python code  
3. Submit pull requests with detailed descriptions

## License

This project is licensed under the MIT License.


### Getting Help
- **Issues**: Report bugs and feature requests via GitHub Issues  