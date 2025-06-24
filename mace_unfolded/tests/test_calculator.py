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
import torch
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
dtype = "float32"
# forward mode autodiff - faster
forward = True
verbose = False
seed = 1234
TARGET_T = 300
n_equil = 10
n_run = 10
model_file_rmax_5 = "/home/swieser/cellar/InAs/wurtzite/training/PBEsol/mace_400/mace_torch_rmax_5/InAs_swa.model"
model_file_rmax_5_64 = "/home/swieser/cellar/InAs/wurtzite/training/PBEsol/mace_400/mace_torch_rmax_5_h64/pot2/InAs_swa.model"
model_file_rmax_4 = "/home/swieser/cellar/InAs/wurtzite/training/PBEsol/mace_400/mace_torch_rmax_4/InAs_swa.model"
model_file_rmax_4_64 = "/home/swieser/cellar/InAs/wurtzite/training/PBEsol/mace_400/mace_torch_rmax_4_h64/InAs_swa.model"
struct_file_conv = "/home/swieser/cellar/InAs/wurtzite/training/PBEsol/mace_400/mace_torch_rmax_5/zb/POSCAR"
# 2x2x2
struct_file_222 = "/home/swieser/cellar/InAs/wurtzite/training/PBEsol/mace_400/mace_torch_rmax_5/222/SPOSCAR"
# 3x3x3
struct_file_333 = "/home/swieser/cellar/InAs/wurtzite/training/PBEsol/mace_400/mace_torch_rmax_5/zb/ph3py/SPOSCAR"
# 4x4x4 - should contain effective cutoff
struct_file_444 = "/home/swieser/cellar/InAs/wurtzite/training/PBEsol/mace_400/mace_torch_rmax_5/444/SPOSCAR"

struct_file = struct_file_222
model_file = model_file_rmax_4_64
flux_dir = "flux_files_unf_-2"
if forward:
    flux_dir += "_fwd"
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
        f"E = {e_pot_per_atom + e_kin_per_atom} eV / atom"
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
unf_calc = UnfoldedHeatFluxCalculator(
    model, device=device, verbose=verbose, forward=forward, dtype=dtype
)


with open(flux_file_name, "w") as hfp:
    hfp.write("Temp c_flux1[1] c_flux1[2] c_flux1[3]\n")
# to output everything
with open(flux_comp_file_name, "w") as hfp:
    hfp.write(
        "Temp force[1] force[2] force[3] pot[1] pot[2] pot[3] int[1] int[2] int[3] conv[1] conv[2] conv[3]\n"
    )

# heat_comp_timer = 0
# heat_out_timer = 0


def compute_heat(current_atoms=atoms, current_integrator=integrator):
    # nonlocal heat_comp_timer, heat_out_timer
    # t1 = time.time()
    if current_integrator.get_number_of_steps() != 1:
        heat_flux = unf_calc.calculate(current_atoms)
        # even though its contribution should be small, it is rather easy to compute
        velocities = current_atoms.get_velocities() * ase.units.fs * 1000
        natoms = current_atoms.get_global_number_of_atoms()
        potential_energies = current_atoms.calc.results["node_energy"]
        unf_pot_energies = unf_calc.results["energies"]
        print(potential_energies)
        for i in range(len(potential_energies)):
            print(potential_energies[i],unf_pot_energies[i])
        print("energy difference:", np.sum(np.abs(potential_energies-unf_pot_energies)))
        print("sum energies:", np.sum(unf_pot_energies))
        print("total energy:", current_atoms.get_potential_energy())
        mmatrix = np.repeat(current_atoms.get_masses(), 3).reshape((natoms, 3))
        kinetic_energies = np.sum((velocities**2) * mmatrix / 2, axis=1)
        atomic_energies = potential_energies + kinetic_energies
        hf_convective_term = (
            velocities.T @ atomic_energies
        ) / current_atoms.get_volume()

        # hf_convective_term = (
        #     np.einsum(
        #         "i,ij->j",
        #         calc.results["node_energy"],
        #         current_atoms.get_velocities() * ase.units.fs * 1000,
        #     )
        #     / current_atoms.get_volume()
        # )

        # t2 = time.time()
        # heat_comp_timer += t2-t1
        flux = heat_flux["heat_flux"]
        pot_term = heat_flux["heat_flux_potential_term"]
        force_term = heat_flux["heat_flux_force_term"]
        # this actually should not be the same thing
        print(heat_flux["forces"][-1])
        print(current_atoms.calc.results["forces"][-1])
        print(np.sum(heat_flux["forces"]-current_atoms.calc.results["forces"]))
        # offset_eng = heat_flux["energies"]-np.min(heat_flux["energies"])
        # offset_eng2 = current_atoms.calc.results["node_energy"]-np.min(current_atoms.calc.results["node_energy"])
        # print(np.sum(offset_eng-offset_eng2))
        # print(current_atoms.calc.results["node_energy"],heat_flux["energies"])
        # print(np.sum(current_atoms.calc.results["node_energy"]-heat_flux["energies"]))
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
trajectory = Trajectory(f"{flux_dir}/test_gk_{TARGET_T}K_.traj", "w", atoms)
nve.attach(trajectory.write, interval=10, current_integrator=nve)
nve.attach(print_md_info, interval=1, current_integrator=nve)
nve.attach(compute_heat, interval=1, current_integrator=nve)

t1 = time.time()
nve.run(n_run)
t_run = time.time() - t1
print("run time:", t_run, flush=True)
# print("heat compute time:", heat_comp_timer, flush=True)
# print("heat output time:", heat_out_timer, flush=True)
