# mace-unfolded

Contains calculators for computing the heat flux for MACE. Additionally, features some convenient command line tools for MACE potentials.

The unfolder was originally written for the publication "Marcel F. Langer, Florian Knoop, Christian Carbogno, Matthias Scheffler, and Matthias Rupp
Phys. Rev. B 108, L100302" and adapted for use with MACE.

## Installation

```bash
pip install .
```

Aside of obvious dependencies, requires the comms package from:

https://github.com/sirmarcel/comms

This is purely for logging and debugging. This dependency might be removed in the future.

It is helpful to use the `cuequivariance` package for acceleration. The code will function without it, but it speeds up the calculation without substantial drawbacks.

## Usage


### Unfolded calculator

The easiest way the unfolded calculator can be used is from a lammps trajectory file. The advantage is that the heat flux does not have to be computed at every timestep.

```bash
perform_mace_green_kubo.py <model_file_name> --flux_dir <output_directory_name> --from_lammps_traj <trajectory_file_name> --pbc F F T --dtype float64
```

The periodic boundary conditions `--pbc` determine in which directions the unfolding is being performed and how many components of the heat flux are calculated and output in files. The script uses a very simple file reader of the lammps dump file. It strictly requires the fields `atom id, element, x, y, z, vx, vy, vz` as the dump contents. There is an alternative more flexible option to use the ASE reader to read the trajectory files using the option `--ase_reader`. However, ASE reads the entire trajectory into the RAM while the internal reader reads the trajectory on-the-fly. Use `--help` for more information regarding the possible options. However, most options are only relevant when directly carrying out the molecular dynamics simulation with an ASE calculator.

With this call the script uses this equation to obtain the potential term of the heat flux:

```math
\mathbf{J}_{\mathrm{pot}} = \sum_{\substack{i\in \mathcal{R}_{\mathrm{cell}} \\ j \in \mathcal{R}_{\mathrm{unf}}}} \frac{\partial\mathbf{B}}{\mathbf{\mathbf{r}}_j} \cdot \mathbf{v}_j - \sum_{j \in \mathcal{R}_{\mathrm{unf}}} \mathbf{r}_j \left( \frac{\partial U}{\partial \mathbf{r}}_j \cdot \mathbf{v}_j \right).
```

Consult [this paper](https://link.aps.org/doi/10.1103/PhysRevB.108.L100302) for a detailed explanation about the potential term. The values will be stored for each evaluated time step in a file called `heat_flux.dat` in the specified `output_directory_name`. The kinetic portion of the flux will also be computed but output seperately in a file `heat_flux_components.dat`.

The script is also capable to perform molecular dynamics directly using an ASE calculator. This helps due to the removed requirement to write the trajectory in a file. But due to the much decreased efficiency of the ASE calculator compared to lammps properly compiled with GPU support and due to the fact that it is typically not required to compute the heat flux every single time step, this is not recommended for actual production runs.

Alternatively, the heat flux calculator can be invoked directly in python to compute the heat flux of any ASE atoms object.

```python
from mace_unfolded.unfolded_heat.unfolder_calculator import UnfoldedHeatFluxCalculator
unf_calc = UnfoldedHeatFluxCalculator(
    MACE_model, device="cuda", forward=False,
)
results = unf_calc.calculate(current_atoms)
potential_flux = results["heat_flux"]
```

Note, that what is obtained here is only the potential flux. For fluids or any system featuring convection it is necessary to also consider the kinetic term. Consult the function `compute_heat` from the `perform_mace_green_kubo` script for the code to compute the full heat flux.


#### Exploiting gauge invariance to reduce the level of noise

The `compute_heat` function also computes the average virial $\sigma_i$ for each atom which allows to subtract the non-contributing parts of the potential heat flux.

```math
\mathbf{J}_\mathrm{gf}(t) = \mathbf{J}_\mathrm{raw} - \frac{1}{V}\sum_i\left< \sigma_i \right>_t\mathbf{v}_i.
```

The atomic virial is computed in a way that its product with the atomic velocities results in the total heat flux at each individual time step. When opening the file `sigma.dat` in the output directory contains the values of $\sigma_i$ for each atom and the number of values depends on the dimensionality of the system. Due to the way the unfolding works, several atomic indices occur multiple times. However, these are effectively just periodic images and for the evaluation of the heat flux contributions the same velocities should be used. Use the script `gauge_fix_flux` to apply the correction to a heat flux file using the velocity from a lammps trajectory. 

For some literature regarding the gauge invariance, see [Ercole et al.](https://pubs.acs.org/doi/10.1021/acs.jctc.9b01174) and for an application of an implementation such as this one, see [Knoop et al.](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.224304).

### MACE heat flux ASE calculator

The calculator that can be found in `mace_unfolded.unfolded_heat.mace_flux_calculator` can be used as an ASE calculator and its primary purpose is to evaluate the heat flux without applying the unfolding of the unit cell. This primarily serves for validation purposes to make sure that the implementation of the unfolded heat flux is correct. The computation of the heat flux is done in three different ways.

In case of a "local" MACE model (number of message passing layers equals to 1) all methods agree including the results from the unfolded calculator. In case of higher numbers of message passing layers, the unfolded solution only aligns with the implementation of the full semi-local heat flux. However, the latter is much slower in terms of computation time.

For more details regarding these equations, see [Langer et al.](https://link.aps.org/doi/10.1103/PhysRevB.108.L100302).

#### Method 1: Full semi-local heat flux

Set `full_semilocal_flux=True` when creating the calculator object.

```math
\mathbf{J}_\mathrm{pot} = \sum_{ijk} \mathbf{r}_{ji} \left[ \left( \frac{\partial U_i}{\partial \mathbf{r}_{kj}} - \frac{\partial U_i}{\partial \mathbf{r}_{jk}} \right) \cdot \mathbf{v}_j \right]
```

Where the indices $i$ and $j$ run over all atoms and $k$ over the neighborhood of atom $j$. This equation is very expensive to evaluate.

#### Method 2:

This implements the equation for the local flux set by `full_local_flux=True` and `full_semilocal_flux=False`.

```math
\mathbf{J}_\mathrm{pot} = \sum_{jk} \mathbf{r}_{kj} \left( \frac{\partial U_j}{\partial \mathbf{r}_{jk}}  \cdot \mathbf{v}_k \right)
```

#### Method 3:

Another local variant set by calling `init_full_local_flux` and `slow_method=True`.

```math
\mathbf{J}_\mathrm{pot} = \sum_{jk} \mathbf{r}_{kj} \left( \frac{\partial U_j}{\partial \mathbf{r}_{k}}  \cdot \mathbf{v}_k \right)
```

### Command line tools

#### Compute phonons with MACE with phonopy

The command line tool to use is `compute_mace_phonons`. Alternatively, the function `mace_unfolded.scripts.compute_mace_phonons.compute_phonons` can be imported and used directly in python. The script interfaces with phonopy and computes the phonons for the `POSCAR` file in the current working directory: 

```bash
compute_mace_phonons $MACE_MODEL_FILE --supercell 2 2 2 --relax
```

If `--relax` is used, `POSCAR_relaxed` is generated with the internal coordinates optimized, copy it to `POSCAR` for further analysis using phonopy with the `force_constants.hdf5` file that was output. With `--ddist` the displacement distance can be set and `--autodiff` computes the force constants analytically with automatic differentiation. Use `--help` for more information regarding the possible options.
