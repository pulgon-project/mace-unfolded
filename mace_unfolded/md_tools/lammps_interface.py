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

import lammps.mliap as lmliap
from lammps import lammps

from ase.atoms import Atoms
from mace_unfolded.md_tools.fileIO import write_lammps_atomic



def from_file(lmp: lammps, struct_file_name: str):
    lmp.cmd.read_data(struct_file_name)


def lmp_cmd_init_from_atoms(lmp:lammps, atoms : Atoms, temp_fname :str = "tmp.lmp"):
    # TODO: write this intelligently. For now write a file and read from it
    # The ASE lammps fileIO is very bad, using a custom version.
    write_lammps_atomic(temp_fname, atoms)
    from_file(lmp, temp_fname)

def init_lammps_mace(
    structure: Atoms | str,
    model_file: str,
    symbols: list,
    mliap: bool = True,
    ngpus: int = 1,
    pbc=[True, True, True],
    kokkos=True,
):
    if kokkos:
        cmdargs = [
            "-k",
            "on",
            "g",
            str(ngpus),
            "-sf",
            "kk",
            "-pk",
            "kokkos",
            "newton",
            "on",
            "neigh",
            "half",
        ]
    else:
        cmdargs = []
        raise NotImplementedError
    lmp = lammps(cmdargs=cmdargs)
    if kokkos:
        lmliap.activate_mliappy_kokkos(lmp)
        # import mliap_unified_couple_kokkos
    else:
        lmliap.activate_mliappy(lmp)
    lmp.cmd.echo("screen")
    lmp.cmd.units("metal")
    lmp.cmd.atom_style("atomic/kk")
    lmp.cmd.atom_modify("map", "yes")
    pbc_mapped = []
    for p in pbc:
        if p:
            pbc_mapped.append("p")
        else:
            pbc_mapped.append("f")
    lmp.cmd.boundary(pbc_mapped[0], pbc_mapped[1], pbc_mapped[2])
    if type(structure) == str:
        from_file(lmp, structure)
    else:
        lmp_cmd_init_from_atoms(lmp, structure)

    if mliap:
        lmp.cmd.pair_style("mliap/kk", "unified", model_file, 0)
        lmp.cmd.pair_coeff("*", "*", *symbols)
    else:
        raise NotImplementedError

    # not sure if this is needed
    lmp.cmd.run_style("verlet/kk")

    # model = lmp.LOADED_MODEL

    return lmp
