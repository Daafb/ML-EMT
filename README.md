# Fractal Particle Opacity Toolkit

Python utilities to compute and store dust opacities and scattering properties for porous fractal particles using Optool and scattnlay.

This project wraps two main backends:
- **Optool** for Mie and MMF opacities and optionally the full scattering matrix
- **scattnlay** for layered sphere scattering and optionally the full scattering matrix

It provides a `Particle` class that:
- derives bulk and fractal properties from a small parameter set
- runs opacity calculations via Optool and scattnlay
- saves results in a robust folder format and can load them back later

## Features

- Fractal particle model based on fractal dimension `D`, prefactor `kf`, monomer radius `a0`, and characteristic radius `Rc`
- Derived quantities: mass, mean density, filling factor, porosity, radius of gyration
- Optional subdivision into `N` radial shells with wavelength dependent effective refractive indices
- Compute opacities:
  - Optool Mie: `RunMie()`
  - Optool MMF: `RunMMF()`
  - scattnlay layered scattering: `RunScatt()`
- Optional scattering matrix output for Optool and scattnlay
- Save and load results safely even if some outputs are missing

## Requirements

Python packages used in the code:
subprocess
shlex
numpy
pandas
matplotlib.pyplot
scattnlay
re
pathlib
json

External dependency:
- **Optool** executable available on your system PATH as `optool`

Notes:
- The code calls `optool` via `subprocess`. Make sure the `optool` command is available in your environment.

## Configuration

Global Optool settings are controlled via:

```python
OPTOOL_SETTINGS = {
    "lmin": 0.1,     # micron
    "lmax": 1000,    # micron
    "nl": 100,
    "method": "mie",
    "NANG": 180
}
