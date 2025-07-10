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

import argparse
import os
import sys
import time

import ase
import ase.io
import numpy as np
from ase.io.trajectory import Trajectory
from ase.atoms import Atoms
from ase.md.verlet import VelocityVerlet
from mace.calculators import MACECalculator
from mace_unfolded.unfolded_heat.unfolder_calculator import UnfoldedHeatFluxCalculator
from mace_unfolded.md_tools.md_utils import (
    print_md_info,
    md_equilibration,
)
import time
from tqdm import tqdm

import subprocess
import copy


class lammps_dump_reader:
    """
    Extremely primitive lammps file reader, memory light but probably not very fast.
    The issue with ASE trajectories: they will be loaded fully into the memory.
    Issue with this: file handle remains opened
    Another issue with this: only supports orthorhombic cells - not true anymore I think
    """

    def __init__(self, filename, num_start=0, unwrap=False):
        self.filename = filename
        self.num_start = num_start
        self.unwrap = unwrap
        self.first_atoms = None

    def __iter__(self):
        self.file_handle = open(self.filename)
        if self.num_start > 0:
            natoms = None
            for i in range(self.num_start):
                if natoms is None:
                    for i in range(3):
                        self.rline()
                    natoms = int(self.rline())
                    for i in range(5 + natoms):
                        self.rline()
                else:
                    for i in range(9 + natoms):
                        self.rline()

        return self

    def rline(self):
        line = self.file_handle.readline()
        if not line:
            raise StopIteration
        return line


    def __next__(self):
        self.rline()
        tstep = int(self.rline())
        self.rline()
        natoms = int(self.rline())
        self.rline()
        cell_block = []
        for i in range(3):
            cell_block.append(np.array(list(map(float, self.rline().split()))))
        # technically it would be nice to parse this. Currently hardcoded ordering: atom id, element, x, y, z, vx, vy, vz
        self.rline()
        elements = []
        self.velocities = np.zeros((natoms, 3))
        self.positions = np.zeros((natoms, 3))
        for i in range(natoms):
            line = self.rline().split()
            elements.append(line[1])
            self.positions[i] = np.array(list(map(float, line[2:5])))
            self.velocities[i] = np.array(list(map(float, line[5:8])))

        cell_matrix = np.zeros((3, 3))
        # why did they have to make this this complicated - just leads to stupid errors
        # they write it is convenient for visualization programs, but it is VERY inconvenient for the human eye
        if len(cell_block[0]) > 2:
            yli = [0, cell_block[2][2]]
            xli = [
                0,
                cell_block[0][2],
                cell_block[1][2],
                cell_block[0][2] + cell_block[1][2],
            ]
        else:
            xli = 0
            yli = 0
        xlo = cell_block[0][0] - np.min(xli)
        xhi = cell_block[0][1] - np.max(xli)
        ylo = cell_block[1][0] - np.min(yli)
        yhi = cell_block[1][1] - np.max(yli)
        zlo = cell_matrix[2][0]
        zhi = cell_block[2][1]
        cell_matrix[0, 0] = xhi - xlo
        cell_matrix[1, 1] = yhi - ylo
        cell_matrix[2, 2] = zhi - zlo
        if len(cell_block[0]) > 2:
            cell_matrix[0, 1] = cell_block[0][2]
            cell_matrix[0, 2] = cell_block[1][2]
            cell_matrix[1, 2] = cell_block[2][2]
        # convert velocities from the very reasonable metal units into those silly ASE units
        self.cell = cell_matrix
        self.symbols = elements
        atoms = Atoms(
            elements,
            self.positions,
            cell=cell_matrix,
            velocities=self.velocities / ase.units.fs / 1000,
        )
        if self.unwrap:
            if self.first_atoms is not None:
                displacement = atoms.get_positions() - self.first_atoms.get_positions()
                frac = np.dot(displacement, np.linalg.inv(self.cell)).round()
                # print(np.sum(frac!=0))
                displacement = np.dot(frac, self.cell)
                self.positions = self.positions - displacement
                atoms.positions = self.positions
            else:
                self.first_atoms = copy.deepcopy(atoms)
        return atoms


def compute_heat(
    current_atoms: ase.Atoms,
    current_integrator,
    unf_calc,
    kine_conversion2eV,
    flux_file_name,
    flux_comp_file_name,
    sigma_dict,
    verbose=False,
):
    # nonlocal heat_comp_timer, heat_out_timer
    # t1 = time.time()
    skip_computation = False
    if current_integrator is not None:
        if current_integrator.get_number_of_steps() == 1:
            skip_computation = True
    if not skip_computation:
        if verbose:
            t1 = time.time()
        temp = current_atoms.get_temperature()
        kine = current_atoms.get_kinetic_energy()
        # print(temp,kine)
        heat_flux = unf_calc.calculate(current_atoms)
        unf_calc.reporter.start("postprocessing")

        unf_calc.reporter.step("computing convective term")
        velocities = current_atoms.get_velocities() * ase.units.fs * 1000
        natoms = current_atoms.get_global_number_of_atoms()
        # potential_energies = current_atoms.calc.results["node_energy"]
        potential_energies = unf_calc.results["energies"]
        mmatrix = np.repeat(current_atoms.get_masses(), 3).reshape((natoms, 3))
        kinetic_energies = np.sum((velocities**2) * mmatrix / 2, axis=1)
        atomic_energies = potential_energies + kinetic_energies * kine_conversion2eV
        hf_convective_term = (
            velocities.T @ atomic_energies
        ) / current_atoms.get_volume()
        # heat_comp_timer += t2-t1
        unf_calc.reporter.step("writing to file")
        flux = heat_flux["heat_flux"]
        pot_term = heat_flux["heat_flux_potential_term"]
        force_term = heat_flux["heat_flux_force_term"]
        with open(flux_file_name, "a") as hfp:
            wstr = "%18.12f " % temp
            for iflux in flux:
                wstr += "%18.12f " % iflux
            hfp.write(wstr + "\n")
            # hfp.write(
            #     "%18.12f %18.12f %18.12f %18.12f\n"
            #     % (current_atoms.get_temperature(), flux[0], flux[1], flux[2])
            # )

        with open(flux_comp_file_name, "a") as hfp:
            wstr = "%18.12f " % temp
            for quantity in [force_term, pot_term, flux, hf_convective_term]:
                for ind in range(len(quantity)):
                    wstr += "%18.12f " % quantity[ind]
            wstr += "\n"
            hfp.write(wstr)

        # sigma processing
        idx = unf_calc.unfolder.unfolding.idx
        full_indices = np.concatenate((np.array(range(len(current_atoms))), idx))
        directions = unf_calc.unfolder.unfolding.directions
        replica_offsets = unf_calc.unfolder.unfolding.replica_offsets
        sigma = unf_calc.results["sigma"]
        if "sigma_full_file_name" in sigma_dict.keys():
            wstr = "\n"
            for i in range(len(sigma)):
                wstr += "%6d " % full_indices[i]
                for j in range(len(sigma[i])):
                    for k in range(len(sigma[i][j])):
                        wstr += "%18.12f " % sigma[i][j][k]
                wstr += "\n"
            with open(sigma_dict["sigma_full_file_name"], "a") as hfp:
                hfp.write(wstr)
        # I guess it makes sense to create some index independent on the ordering. Using idx plus directions? I think replica offsets should not be so important here.
        uind = idx + len(current_atoms) * directions
        if "uinds" not in sigma_dict.keys():
            sigma_dict["uinds"] = uind
        else:
            if len(sigma_dict["uinds"]) != len(uind):
                sigma_dict["uinds"] = np.append(
                    sigma_dict["uinds"], uind[~np.isin(uind, sigma_dict["uinds"])]
                )
                # print("newinds", len(sigma_dict["uinds"]))
            elif not np.allclose(sigma_dict["uinds"], uind):
                sigma_dict["uinds"] = np.append(
                    sigma_dict["uinds"], uind[~np.isin(uind, sigma_dict["uinds"])]
                )
                # print("newinds", len(sigma_dict["uinds"]))
        actinds = np.concatenate(
            (
                np.array(range(len(current_atoms))),
                np.array([np.where(sigma_dict["uinds"] == i)[0][0] for i in uind])
                + len(current_atoms),
            )
        )
        # print(len(current_atoms),len(actinds))
        if "N" not in sigma_dict.keys():
            sigma_dict["N"] = np.ones(len(sigma))
        else:
            sigma_dict["N"] += 1

        if "sigma_sum" not in sigma_dict.keys():
            sigma_dict["sigma_sum"] = sigma
        else:
            lendiff = (len(current_atoms) + len(sigma_dict["uinds"])) - len(
                sigma_dict["sigma_sum"]
            )
            if lendiff != 0:
                sigma_dict["sigma_sum"] = np.concatenate(
                    (
                        sigma_dict["sigma_sum"],
                        np.zeros(
                            (
                                lendiff,
                                sigma_dict["sigma_sum"].shape[1],
                                sigma_dict["sigma_sum"].shape[2],
                            )
                        ),
                    ),
                    axis=0,
                )
                sigma_dict["N"] = np.append(sigma_dict["N"], np.ones(lendiff))
            sigma_dict["sigma_sum"][actinds] += sigma

        if sigma_dict["N"][0] % sigma_dict["sigma_output_freq"] == 0:
            all_inds = np.concatenate(
                (np.array(range(len(current_atoms))), sigma_dict["uinds"])
            )
            with open(sigma_dict["sigma_file_name"], "w") as hfp:
                sigma_avg = (
                    sigma_dict["sigma_sum"] / sigma_dict["N"][:, np.newaxis, np.newaxis]
                )
                wstr = "\n"
                for i in range(len(sigma_avg)):
                    wstr += "%6d " % (all_inds[i] % len(current_atoms))
                    for j in range(len(sigma_avg[i])):
                        for k in range(len(sigma_avg[i][j])):
                            wstr += "%18.12f " % sigma_avg[i][j][k]
                    wstr += "\n"
                hfp.write(wstr)
        unf_calc.reporter.done()
        if verbose:
            print("Full unfolding time: ", time.time() - t1)
        # heat_out_timer += time.time()-t2


def time_tracker(current_atoms, current_integrator, prev_time):
    t1 = time.time()
    time_diff = t1 - prev_time[0]
    print("Time for step: ", time_diff)
    prev_time[0] = t1
    return prev_time


def create_flux_files(flux_file_name, flux_comp_file_name, sigma_dict):
    with open(flux_file_name, "w") as hfp:
        hfp.write("Temp c_flux1[1] c_flux1[2] c_flux1[3]\n")
    with open(flux_comp_file_name, "w") as hfp:
        hfp.write(
            "Temp force[1] force[2] force[3] pot[1] pot[2] pot[3] int[1] int[2] int[3] conv[1] conv[2] conv[3]\n"
        )
    if "unfolded_atoms_file_name" in sigma_dict.keys():
        with open(sigma_dict["unfolded_atoms_file_name"], "w") as hfp:
            hfp.write("index idx direction roffset[1] roffset[2] roffset[3]\n")
    # if "sigma_file_name" in sigma_dict.keys():
    #     with open(sigma_dict["sigma_file_name"], "w") as hfp:
    #         hfp.write("# index sigma\n")
    if "sigma_full_file_name" in sigma_dict.keys():
        with open(sigma_dict["sigma_full_file_name"], "w") as hfp:
            hfp.write("# index sigma\n")


def perform_GK_simulation(
    model_file,
    struct_file="POSCAR",
    seed=1234,
    dt=1.0,
    temperature=300,
    n_equil=10000,
    n_run=1000000,
    PBC=[True, True, True],
    device="cuda",
    flux_dir="flux_files_unf",
    dtype="float32",
    restart=False,
    traj_output_interval=100,
    verbose=False,
    from_lammps_traj=None,
    ase_reader=False,
    full_sigma=False,
):
    # argparse is not good enough for booleans
    if isinstance(PBC[0], str):
        PBC = [x.lower() == "true" or x.lower() == "t" for x in PBC]
    T_limit = temperature * 5

    flux_file_name = f"{flux_dir}/heat_flux.dat"
    flux_comp_file_name = f"{flux_dir}/heat_flux_components.dat"
    traj_file_name = f"{flux_dir}/test_equil_{temperature}K_.traj"
    sigma_dict = {}
    sigma_dict["sigma_file_name"] = f"{flux_dir}/sigma.dat"
    sigma_dict["unfolded_atoms_file_name"] = f"{flux_dir}/unfolded_atoms.dat"
    sigma_dict["sigma_output_freq"] = 200
    if full_sigma:
        sigma_dict["sigma_full_file_name"] = f"{flux_dir}/sigma_full.dat"

    if from_lammps_traj is not None:
        if restart:
            cmd = f"wc -l {flux_file_name}"
            read_from_num = (
                int(
                    subprocess.run(cmd, shell=True, capture_output=True)
                    .stdout.decode()
                    .split()[0]
                )
                - 1
            )
        else:
            read_from_num = 0

        if ase_reader:
            traj = ase.io.read(from_lammps_traj, f"{read_from_num}:")
        else:
            traj = lammps_dump_reader(from_lammps_traj, read_from_num)
    else:
        traj = None

    if not os.path.exists(flux_dir):
        os.mkdir(flux_dir)
    try:
        calc = MACECalculator(
            model_file, device=device, default_dtype=dtype, enable_cueq=True
        )
    except:
        print("cuequivariance acceleration not available. Running conventionally.")
        calc = MACECalculator(model_file, device=device, default_dtype=dtype)
    if traj is None:
        if verbose and "verbose" in calc.__dir__():
            calc.verbose = True
        # model = torch.load(model_file, map_location=device)
        # model.to(device)
        equilibrate = True
        rng = np.random.default_rng(seed=seed)
        if restart:
            trajectory = Trajectory(traj_file_name, "r")
            num_entries = len(trajectory)
            # the first structure is also stored in the trajectory
            expected_equil_entries = n_equil // traj_output_interval + 1
            if num_entries >= expected_equil_entries:
                number_file_lines = sum(1 for line in open(flux_file_name))
                number_comp_lines = sum(1 for line in open(flux_comp_file_name))
                restart_from = (
                    number_file_lines // traj_output_interval
                ) * traj_output_interval
                to_remove = number_file_lines - restart_from
                print(
                    f"actually finished lines: {number_file_lines}. Restarting from {restart_from}"
                )
                print(f"removing {to_remove} entries from flux files")
                cmd = f"sed '{restart_from+2}, $ d' {flux_file_name} -i"
                os.system(cmd)
                number_comp_lines = sum(1 for line in open(flux_comp_file_name))
                restart_from_comp = (
                    number_comp_lines // traj_output_interval
                ) * traj_output_interval
                cmd = f"sed '{restart_from_comp+2}, $ d' {flux_comp_file_name} -i"
                os.system(cmd)

                equilibrate = False
                n_run = n_run - restart_from
                print(f"running remaining {n_run} NVE steps")
            else:
                n_equil = n_equil - (num_entries - 1) * traj_output_interval
                print(f"running remaining {n_equil} NVT equilibration steps")
            atoms = trajectory[-1]
            # when doing multiple restarts, the number of steps to be done might be slighly off
            # however that is such a tiny thing and ultimately inconsequential
            trajectory = Trajectory(traj_file_name, "a", atoms)
            init_vel = False

        else:

            atoms = ase.io.read(struct_file)
            init_vel = True
            trajectory = Trajectory(traj_file_name, "w", atoms)
        atoms.pbc = PBC
        atoms.calc = calc

        if equilibrate:
            md_equilibration(
                atoms,
                temperature=temperature,
                dt=dt,
                n_equil=n_equil,
                T_limit=T_limit,
                traj_output_interval=traj_output_interval,
                rng=rng,
                trajectory=trajectory,
                init_velocities=init_vel,
            )

            # only make new files if run with equilibration
            create_flux_files(
                flux_file_name=flux_file_name,
                flux_comp_file_name=flux_comp_file_name,
                sigma_dict=sigma_dict,
            )

    ### HEAT FLUX

    # calculator can be run in ensemble mode, the unfolded one not yet
    model = calc.models[0]
    if len(calc.models) > 1:
        print(
            "WARNING: ensemble mode not yet supported for the UnfoldedHeatFluxCalculator"
        )
    unf_calc = UnfoldedHeatFluxCalculator(
        model, device=device, verbose=verbose, dtype=dtype, forward=False, pbc=PBC
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

    # main evaluation
    if traj is None:
        nve = VelocityVerlet(
            atoms,
            dt * ase.units.fs,
        )
        # trajectory = Trajectory(f"{flux_dir}/test_gk_{TARGET_T}K_.traj", "w", atoms)
        nve.attach(
            trajectory.write, interval=traj_output_interval, current_integrator=nve
        )
        nve.attach(
            print_md_info, interval=1, current_atoms=atoms, current_integrator=nve
        )
        nve.attach(
            compute_heat,
            interval=1,
            current_atoms=atoms,
            current_integrator=nve,
            unf_calc=unf_calc,
            kine_conversion2eV=kine_conversion2eV,
            flux_file_name=flux_file_name,
            flux_comp_file_name=flux_comp_file_name,
            sigma_dict=sigma_dict,
            verbose=verbose,
        )
        pt = np.array([time.time()])
        if verbose:
            nve.attach(
                time_tracker,
                interval=1,
                current_atoms=atoms,
                current_integrator=nve,
                prev_time=pt,
            )

        t1 = time.time()
        nve.run(n_run)
        t_run = time.time() - t1
        print("run time:", t_run, flush=True)
        # print("heat compute time:", heat_comp_timer, flush=True)
        # print("heat output time:", heat_out_timer, flush=True)

    else:
        if not restart:
            create_flux_files(
                flux_file_name=flux_file_name,
                flux_comp_file_name=flux_comp_file_name,
                sigma_dict=sigma_dict,
            )
        for atoms in tqdm(traj):

            compute_heat(
                current_atoms=atoms,
                current_integrator=None,
                unf_calc=unf_calc,
                kine_conversion2eV=kine_conversion2eV,
                flux_file_name=flux_file_name,
                flux_comp_file_name=flux_comp_file_name,
                sigma_dict=sigma_dict,
                verbose=verbose,
            )


def main():
    parser = argparse.ArgumentParser(
        description="perform a Green-Kubo simulation for a MACE model using ASE as the MD backend. Alternatively, read from a LAMMPS trajectory"
    )

    parser.add_argument(
        "--struct",
        dest="struct_file",
        type=str,
        default="POSCAR",
        help="starting geometry for the MD run",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=1234,
        help="random seed for the MD run",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=300,
        help="target temperature for the MD run in K",
    )
    parser.add_argument(
        "--dt",
        dest="dt",
        type=float,
        default=1.0,
        help="time step of simulation in fs",
    )
    parser.add_argument(
        "--n_equil",
        dest="n_equil",
        type=int,
        default=20000,
        help="number of equilibration steps",
    )
    parser.add_argument(
        "--n_run",
        dest="n_run",
        type=int,
        default=1000000,
        help="number of production steps",
    )
    parser.add_argument(
        "--pbc",
        dest="PBC",
        type=str,
        nargs=3,
        default=[True, True, True],
        help="PBC for the MD run",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="cuda",
        help="device for the MD run",
    )
    parser.add_argument(
        "--flux_dir",
        dest="flux_dir",
        type=str,
        default="flux_files_unf",
        help="directory for the flux files",
    )
    parser.add_argument(
        "--dtype",
        dest="dtype",
        type=str,
        default="float32",
        help="data type for the MD run",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        default=False,
        help="restart the MD run",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="output detailed computation time breakdown",
    )
    parser.add_argument(
        "--from_lammps_traj",
        type=str,
        default=None,
        help="read from lammps trajectory instead of evaluating MD directly with ASE",
    )
    parser.add_argument(
        "--ase_reader",
        dest="ase_reader",
        action="store_true",
        default=False,
        help="enable ASE reader",
    )
    parser.add_argument(
        "--full_sigma",
        dest="full_sigma",
        action="store_true",
        default=False,
        help="output sigma step-by-step",
    )

    parser.add_argument("model_file", help="file name for the model parameters")

    args = parser.parse_args()

    afp = open("p_cmnd.log", "a")
    afp.write(" ".join(sys.argv) + "\n")
    afp.close()

    perform_GK_simulation(**vars(args))


if __name__ == "__main__":
    main()
