#!/usr/bin/env python

# Copyright 2025 The PULGON Project Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from mace_unfolded.unfolded_heat.unfolder_calculator import UnfoldedHeatFluxCalculator

import ase
import ase.io
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
import numpy as np
from ase.io.trajectory import Trajectory
import time
import os


device = "cuda"
seed = 1234
TARGET_T = 300
n_equil = 1000
n_run = 100000
model_file = "InAs_swa_local.model"
struct_file = "POSCAR"
dtype="float32"

flux_dir = "flux_files_unf"
if not os.path.exists(flux_dir):
    os.mkdir(flux_dir)

flux_file_name = f"{flux_dir}/heat_flux.dat"
flux_comp_file_name = f"{flux_dir}/heat_flux_components.dat"
atoms = ase.io.read(struct_file)


from mace.calculators import MACECalculator

calc = MACECalculator(model_file, device=device, default_dtype=dtype)
# model = torch.load(model_file, map_location=device)
# model.to(device)
rng = np.random.default_rng(seed=seed)
MaxwellBoltzmannDistribution(atoms, temperature_K=float(TARGET_T), rng=rng)
atoms.calc = calc


integrator = Langevin(
    atoms,
    1.0 * ase.units.fs,
    temperature_K=float(TARGET_T),
    friction=2e-2,
    rng=rng,
)


def print_md_info(current_atoms=atoms, current_integrator=integrator):
    n_atoms = len(current_atoms)
    e_pot_per_atom = current_atoms.get_potential_energy() / n_atoms
    e_kin_per_atom = current_atoms.get_kinetic_energy() / n_atoms
    kinetic_temperature = e_kin_per_atom / (1.5 * ase.units.kB)
    print(
        f"Step #{current_integrator.nsteps + 1}: "
        f"Epot = {e_pot_per_atom} eV / atom, T = {kinetic_temperature} K, "
        f"E = {e_pot_per_atom + e_kin_per_atom} eV / atom",flush=True
    )


integrator.attach(print_md_info, interval=1)

t1 = time.time()
integrator.run(n_equil)
t_equil = time.time() - t1
print("equilibration time:", t_equil, flush=True)


### HEAT FLUX

# calculator can be run in ensemble mode
model = calc.models[0]
# it probably makes sense to rewrite this unfolded calculator to act as a normal ASE calculator but for now this should work as well
unf_calc = UnfoldedHeatFluxCalculator(model, device=device, verbose=False, dtype=dtype)


with open(flux_file_name, "w") as hfp:
    hfp.write("Temp c_flux1[1] c_flux1[2] c_flux1[3]\n")
# to output everything
with open(flux_comp_file_name, "w") as hfp:
    hfp.write(
        "Temp force[1] force[2] force[3] pot[1] pot[2] pot[3] int[1] int[2] int[3] conv[1] conv[2] conv[3]\n"
    )

# heat_comp_timer = 0
# heat_out_timer = 0
    
eV2J = 1.60218e-19
J2eV = 1.0 / eV2J
amu2kg = 1.66054e-27
A2m = 1.0e-10
ps2s = 1.0e-12
kine2J = amu2kg * A2m**2 / ps2s**2
kine_conversion2eV = kine2J * J2eV

def compute_heat(current_atoms=atoms, current_integrator=integrator):
    # nonlocal heat_comp_timer, heat_out_timer
    # t1 = time.time()
    if current_integrator.get_number_of_steps() != 0:
        heat_flux = unf_calc.calculate(current_atoms)
        velocities = current_atoms.get_velocities() * ase.units.fs * 1000
        potential_energies = heat_flux["energies"]
        kinetic_energies = np.einsum("i,ij->i", current_atoms.get_masses(), velocities**2)/2
        atomic_energies = potential_energies + kinetic_energies * kine_conversion2eV
        hf_convective_term = (
            velocities.T @ atomic_energies
        ) / current_atoms.get_volume()
        
        # t2 = time.time()
        # heat_comp_timer += t2-t1
        flux = heat_flux["heat_flux"]
        pot_term = heat_flux["heat_flux_potential_term"]
        force_term = heat_flux["heat_flux_force_term"]
        with open(flux_file_name, "a") as hfp:
            hfp.write(
                "%18.12f %18.12f %18.12f %18.12f\n"
                % (current_atoms.get_temperature(), flux[0], flux[1], flux[2])
            )
        with open(flux_comp_file_name, "a") as hfp:
            wstr = "%18.12f " % current_atoms.get_temperature()
            for quantity in [force_term, pot_term, flux, hf_convective_term]:
                for ind in range(3):
                    wstr += "%18.12f " % quantity[ind]
            wstr += "\n"
            hfp.write(wstr)
        # heat_out_timer += time.time()-t2


nve = VelocityVerlet(
    atoms,
    1.0 * ase.units.fs,
)
trajectory = Trajectory(f"{flux_dir}/GK.traj", "w", atoms)
nve.attach(trajectory.write, interval=100, current_integrator=nve)
nve.attach(print_md_info, interval=1, current_integrator=nve)
nve.attach(compute_heat, interval=1, current_integrator=nve)

t1 = time.time()
nve.run(n_run)
t_run = time.time() - t1
print("run time:", t_run, flush=True)
# print("heat compute time:", heat_comp_timer, flush=True)
# print("heat output time:", heat_out_timer, flush=True)
