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

from mace_unfolded.unfolded_heat.mace_flux_calculator import MACE_flux_calculator

import numpy as np

import ase
import ase.io
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory

import time
import os

TARGET_T = 300
seed = 1234
n_equil = 100
n_run = 100
model_file= "InAs_swa_local.model"
device = "cuda"


struct_file_222 = "POSCAR"
flux_dir = "flux_files_full_local_2"
if not os.path.exists(flux_dir):
    os.mkdir(flux_dir)
atoms = ase.io.read(struct_file_222)

flux_file_name = f"{flux_dir}/heat_flux.dat"
flux_comp_file_name = f"{flux_dir}/heat_flux_components.dat"

calc = MACE_flux_calculator(model_paths=model_file, device=device)

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
        f"E = {e_pot_per_atom + e_kin_per_atom} eV / atom"
    )


integrator.attach(print_md_info, interval=1)

t1 = time.time()
integrator.run(n_equil)
t_equil = time.time() - t1
print("equilibration time:", t_equil, flush=True)

# can turn on flux calculation
calc.init_full_local_flux()

with open(flux_file_name, "w") as hfp:
    hfp.write("Temp c_flux1[1] c_flux1[2] c_flux1[3]\n")
# to output everything
with open(flux_comp_file_name, "w") as hfp:
    hfp.write(
        "Temp pot[1] pot[2] pot[3] conv[1] conv[2] conv[3] full[1] full[2] full[3]\n"
    )
def print_heat(current_atoms=atoms, current_integrator=integrator):

    # don't output first step - ASE will use the last step of the equilibration run in NVE
    # and then the heat flux is not computed since it was not done during NVE.
    if current_integrator.get_number_of_steps() != 0:

        flux_pot = current_atoms.calc.results["heat_flux_pot"]
        flux_conv = current_atoms.calc.results["heat_flux_conv"]
        flux_sum = flux_pot + flux_conv

        with open(flux_file_name, "a") as hfp:
            hfp.write(
                "%18.12f %18.12f %18.12f %18.12f\n"
                % (current_atoms.get_temperature(), flux_pot[0], flux_pot[1], flux_pot[2])
            )
        with open(flux_comp_file_name, "a") as hfp:
            wstr = "%18.12f " % current_atoms.get_temperature()
            for quantity in [flux_pot, flux_conv, flux_sum]:
                for ind in range(3):
                    wstr += "%18.12f " % quantity[ind]
            wstr += "\n"
            hfp.write(wstr)


nve = VelocityVerlet(
    atoms,
    1.0 * ase.units.fs,
)

# trajectory = Trajectory(f"{flux_dir}/test_gk_{TARGET_T}K_.traj", "w", atoms)
# nve.attach(trajectory.write, interval=10, current_integrator=nve)
nve.attach(print_md_info, interval=1, current_integrator=nve)
nve.attach(print_heat, interval=1, current_integrator=nve)
# here then, we only would need to write it
#nve.attach(compute_heat, interval=1, current_integrator=integrator)

t1 = time.time()
nve.run(n_run)
t_run = time.time() - t1
print("run time:", t_run, flush=True)
