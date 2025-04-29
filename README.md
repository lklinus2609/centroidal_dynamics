# Momentum-Based Balance Controller for Unitree G1

This repository implements the momentum-based balance controller described in the paper "A momentum-based balance controller for humanoid robots on non-level and non-stationary ground" by Lee & Goswami. The controller is implemented for the Unitree G1 humanoid robot.

## Prerequisites

- Ubuntu 24.04 LTS
- Python 3.10+
- MuJoCo 3.0.0+
- Pinocchio 2.6.20+

## Installation

1. Install MuJoCo and Pinocchio:

   ```bash
   # Install MuJoCo
   pip install mujoco
   
   # Install Pinocchio
   sudo apt-add-repository ppa:robotpkg/ppa
   sudo apt update
   sudo apt install robotpkg-py3-pinocchio


2. Install other dependencies:

    