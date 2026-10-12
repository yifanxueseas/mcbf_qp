# MCBF-QP: Modulated Control Barrier Function Quadratic Programs

This repository provides an implementation of **On-Manifold Modulated Control Barrier Function Quadratic Programs (MCBF-QPs)** for safe and liveness-aware obstacle avoidance. By integrating geometric guidance from Modulation of Dynamical Systems (Mod-DS) into CBF-QPs, On-Manifold MCBF-QP combines the formal safety guarantees and control-affine generality of CBF-QPs with the liveness capabilities of Mod-DS, promoting goal convergence while avoiding undesirable equilibria and dysfunctional circulation.

The framework supports safe and liveness-aware robot navigation around arbitrarily shaped concave obstacles in static and dynamic environments.

**Paper:** *No Undesirable Equilibria, No Collisions: Combining Modulation and Control Barrier Functions for Liveness and Safety in Dynamic Collision Avoidance*

**Authors:** Yifan Xue and Nadia Figueroa, University of Pennsylvania.

**Code Contributors:** Yifan Xue and Ze Zhang.

**Project Website:** [MCBF-QP](https://yifanxueseas.github.io/mcbf_web/)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yifanxueseas/mcbf_qp.git
   cd mcbf_qp
   ```

2. **Create the Conda environment:**

   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment:**

   ```bash
   conda activate mcbf_controller
   ```

## Configuration

The proposed MCBF-QP and standard CBF controllers use a linear dynamical system (DS) as the nominal controller and support both differential-drive (unicycle model) and omnidirectional-drive robots.

Controller and environment settings can be configured in `config/env_controller.yaml`.

## Testing and Visualization

**1. Test the GPDF-based distance field generator**

   ```bash
   cd src/
   python test_gpdf_image.py
   ```

**2. Run the real-time visualization and debugger for MCBF-QP controllers**

   ```bash
   cd src/
   python ctrl_omcbf_debugger.py
   ```

**3. Generate GIFs of multiple robots navigating in a shared environment using MCBF-QP controllers**

   ```bash
   cd src/
   python test_ctrl_omcbf_gif.py
   ```

## Customize Your Own Environments

- **Dynamic human motion:** Defined in `src/pkg_moving_object/human_trajs.py`. Modify this file to customize human trajectories.
- **Static environment infrastructure:** Defined in the `__init__` method of the `Env` class in `src/basic_boundary_function/env.py`. Modify this method to customize the static environment layout.