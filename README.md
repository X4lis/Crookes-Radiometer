# Crookes Radiometer — Multiphysics Simulation

A simulation of a Crookes radiometer that models nine distinct physical mechanisms simultaneously: radiative heating, thermal conduction through the vane, thermal creep (the dominant drive force in the partial-vacuum regime), differential molecular rebound pressure, photon radiation pressure, rigid-body rotation, viscous drag, buoyancy-driven convective torque, and electrostatic charging of the glass bulb.

Two modes are available — a headless batch simulation (`crookes_radiometer.py`) that outputs plots and a pressure-sweep curve, and a real-time interactive version (`crookes_interactive.py`) built with pygame where you can drag the light source and heat spots, scroll to change the gas pressure, and watch the torque contributions update live.

A write-up of the physical model, numerical methods, and results is in [paper/crookes_multiphysics_paper.pdf](paper/crookes_multiphysics_paper.pdf).

## Why thermal creep and not radiation pressure?

The common misconception is that the vanes spin because light pushes on them directly. Radiation pressure actually pushes the *white* (reflective) face harder than the black face, which would make the radiometer spin the wrong way, and the force is far too small to explain the observed speeds anyway.

The real mechanism is thermal creep (also called the thermophoretic edge effect or the Knudsen force). Gas molecules near the hot edge of the black face pick up extra thermal velocity and bounce off with greater momentum, producing a net reaction force. This only works efficiently when the mean free path of the gas molecules is comparable to the size of the vanes — the transition regime around Kn ≈ 1. Too much gas (atmospheric pressure) and viscosity kills the spin; too little and there are not enough molecules to transfer momentum. The sweet spot is roughly 1–100 Pa, which is why commercial radiometers are pumped down to a partial vacuum.

## Physics modelled

| # | Effect | Regime |
|---|--------|--------|
| 1 | Radiative heating → vane temperature field | all |
| 2 | 1-D thermal conduction across vane thickness | all |
| 3 | Thermal creep (LGE / Knudsen force) | Kn ~ 1 |
| 4 | Differential molecular rebound pressure | Kn >> 1 |
| 5 | Photon radiation pressure | all (tiny) |
| 6 | Rigid-body rotation, moment of inertia | all |
| 7 | Viscous drag (free-molecular + continuum) | all |
| 8 | Buoyancy-driven convective torque + glass wall attenuation | Kn << 1 |
| 9 | Electrostatic image-charge torque on metallised vanes | all |

## Files

```
crookes_radiometer.py     — physics engine + batch simulation + matplotlib output
crookes_interactive.py    — real-time pygame visualisation
paper/                    — write-up
```

## Usage

**Batch simulation** (generates plots, no display needed):
```bash
python crookes_radiometer.py
```

**Interactive simulation**:
```bash
python crookes_interactive.py
```

Controls for the interactive mode:

| Input | Action |
|-------|--------|
| Drag ☀ sun icon | Move the collimated light beam |
| Drag 🔥 heat spots | Reposition point heat sources |
| Scroll on the bulb | Change gas pressure (shifts the Kn regime) |
| `L` | Toggle light on / off |
| `+` / `-` | Irradiance up / down |
| `E` | Cycle electrostatic charge (none / +1 nC / +5 nC / −1 nC) |
| `Space` | Pause / resume |
| `R` | Reset to defaults |

## Dependencies

```
numpy
matplotlib
pygame
```

Install with `pip install numpy matplotlib pygame`.

## Paper

The full derivation of each torque model, the numerical integration scheme, and simulation results are in [paper/crookes_multiphysics_paper.pdf](paper/crookes_multiphysics_paper.pdf).

## Citation

If you use this work in your research, please cite it as:

```bibtex
@software{crookes_radiometer,
  title  = {A Multiphysics Simulation of the Crookes Radiometer with Real-Time Interactive Visualisation},
  author = {Phoenix Avila},
  year   = {2026},
  url    = {https://doi.org/10.5281/zenodo.20572580},
  doi    = {10.5281/zenodo.20572580}
}
```

Or in plain text:

> Phoenix Avila (2026). *A Multiphysics Simulation of the Crookes Radiometer with Real-Time Interactive Visualisation*. Zenodo. https://doi.org/10.5281/zenodo.20572580

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20572580.svg)](https://doi.org/10.5281/zenodo.20572580)
