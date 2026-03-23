# RL-Matheuristics-for-QOT-RSA

**Note:** This folder contains code extracted from an IPython notebook using a GenAI tool.

This repository contains the modularized code for the thesis work on Reinforcement Learning enhanced Matheuristics for Quality of Transmission aware Routing and Spectrum Allocation (QoT-RSA) in optical networks.

## Structure

- `hardware_setup.py`: Hardware calibration and device setup for PyTorch.
- `optical_physics.py`: Optical physics calculations, QoT preprocessing, topology generation, demand generation, and ILP model building.
- `matheuristics.py`: State-of-the-art matheuristic algorithms for RSA.
- `rl_components.py`: Reinforcement Learning components including GNN models, training loop, benchmarking, and reporting.
- `plotting_utils.py`: Plotting and visualization utilities.
- `main.py`: Main driver script to run the full pipeline (training and benchmarking).
- `latex_assets/`: Directory for generated plots, CSVs, and model weights.

## Usage

To run the complete pipeline:

```bash
python main.py
```

This will train RL agents on training topologies and benchmark them on test topologies, generating all necessary assets for the thesis.

## Dependencies

- PyTorch
- PySCIPOpt
- NetworkX
- Pandas
- Matplotlib
- NumPy
- Torch Geometric

Install via pip or conda as appropriate for your environment.
