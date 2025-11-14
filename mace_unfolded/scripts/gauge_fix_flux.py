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

import numpy as np

# ASE is too slow
from mace_unfolded.md_tools.fileIO import lammps_dump_reader

from tqdm import tqdm
import ase.units

import argparse


def perform_fix(
    flux_file,
    trajectory_file,
    sigma_avg_file,
    fixed_flux_file,
    invariant_terms_file=None,
    num_dim=1,
    convective=False,
):
    """
    currently implemented for the 1D case
    """
    sigma_data = np.loadtxt(sigma_avg_file, skiprows=1)
    heat_flux = np.loadtxt(flux_file, skiprows=1)
    temperatures = heat_flux[:, 0]
    if convective:
        # full potential term plus convective flux
        heat_flux = (
            heat_flux[:, (num_dim * 2 + 1) : (num_dim * 3 + 1)]
            + heat_flux[:, -num_dim:]
        )
    else:
        heat_flux = heat_flux[:, 1:]
    indices = np.array(list(map(int, sigma_data[:, 0])))
    sigma = np.zeros((len(indices), num_dim, 3))
    for i in range(num_dim):
        sigma[:, i, :] = sigma_data[:, (i * 3 + 1) :]
    # max_N = 567
    traj = lammps_dump_reader(trajectory_file, 0)
    invariant_terms = []
    with open(fixed_flux_file, "w") as hfp:
        hfp.write("# Temp flux\n")
        entry_id = 0
        for atoms in tqdm(traj):

            velocities = atoms.get_velocities()
            # nw = Nanowire(atoms)
            num_regular = len(atoms)
            num_unfolded = len(indices) - num_regular
            # extended_indices = np.concatenate((range(num_regular), indices))
            extended_indices = indices
            extended_velocities = velocities[extended_indices] * ase.units.fs * 1000
            # don't use nanowire volume here - correction will be applied later in the cepstral analysis script
            gauge_invariant_term = (
                np.einsum("ijk,ik->j", sigma, extended_velocities) / atoms.get_volume()
            )
            invariant_terms.append(gauge_invariant_term)
            gauge_fixed_flux = heat_flux[entry_id] - gauge_invariant_term
            wstr = "%18.12f " % temperatures[entry_id]
            for i in range(num_dim):
                wstr += "%18.12f " % gauge_fixed_flux[i]
            wstr += "\n"
            hfp.write(wstr)
            entry_id += 1
    if invariant_terms_file is not None:
        np.savetxt(invariant_terms_file, invariant_terms)


def main():
    parser = argparse.ArgumentParser(
        description="perform gauge fixing from an existing LAMMPS trajectory and using the average sigma values output from the existing unfolded heat flux calculation"
    )

    parser.add_argument(
        "--sigma",
        dest="sigma_avg_file",
        type=str,
        default="flux_files_unf/sigma.dat",
        help="averaged sigma values from the MD run",
    )

    parser.add_argument(
        "--flux",
        dest="flux_file",
        type=str,
        default="flux_files_unf/heat_flux.dat",
        help="file containing the flux values (full flux, including convective part) from the MD run",
    )

    parser.add_argument(
        "--trajectory",
        dest="trajectory_file",
        type=str,
        default="dump.vel",
        help="LAMMPS trajectory file containing the velocities",
    )

    parser.add_argument(
        "--fixed_flux",
        dest="fixed_flux_file",
        type=str,
        default="flux_files_unf/heat_flux_gauge_fixed.dat",
        help="output file name of the gauge fixed flux",
    )

    parser.add_argument(
        "--invariant_terms",
        dest="invariant_terms_file",
        type=str,
        default=None,
        help="output file name of the invariant flux contribution",
    )

    parser.add_argument(
        "--num_dim",
        dest="num_dim",
        type=int,
        default=1,
        help="number of dimensions of the flux",
    )
    parser.add_argument(
        "--convective",
        dest="convective",
        action="store_true",
        default=False,
        help="Also use convective part of the flux, the heat flux file must be the components file. It is recommended to use the full flux for gauge fixing.",
    )

    args = parser.parse_args()

    perform_fix(**vars(args))


if __name__ == "__main__":
    main()
