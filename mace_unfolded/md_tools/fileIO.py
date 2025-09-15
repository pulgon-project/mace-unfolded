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
from ase.atoms import Atoms
import copy
import ase


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
        # It would be nice to parse this. Currently hardcoded ordering: atom id, element, x, y, z, vx, vy, vz
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
        # convert velocities from metal units into ASE units
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
                displacement = np.dot(frac, self.cell)
                self.positions = self.positions - displacement
                atoms.positions = self.positions
            else:
                self.first_atoms = copy.deepcopy(atoms)
        return atoms


def write_lammps_atomic(
    fname, atoms: Atoms, atom_types=None, charges=None, bond_num=None, uelemsort=False
):
    # only write out coordinates without any extra information about bonds dihedrals etc.
    # masses can be added by hand for each element type - otherwise they will automatically
    # be assigned by a table
    # separate atom types can be defined as well
    # if charges are specified - the file type will be "charge" instead
    # if bond_num is given - space for this many bond types will be reserved

    cell = atoms.get_cell().array
    elems = atoms.get_chemical_symbols()
    coors = atoms.positions
    masses = atoms.get_masses()

    with open(fname, "w") as fp:
        fp.write(" # generated by swscripts\n\n")
        fp.write("%12d %s\n" % (len(coors), "atoms"))

        if atom_types is None:
            uelems, uinds = np.unique(elems, return_index=True)
        else:
            uelems, uinds = np.unique(atom_types, return_index=True)
        umasses = masses[uinds]
        if uelemsort:
            logical = np.argsort(uelems)
        else:
            logical = np.argsort(uinds)
        uelems = uelems[logical]
        umasses = umasses[logical]
        fp.write("%12d %s\n" % (len(uelems), "atom types"))
        if bond_num is not None:
            fp.write("%12d %s\n\n" % (bond_num, "bond types"))
        fp.write("\n")
        assert (
            (cell[0, 1] == 0) & (cell[0, 2] == 0) & (cell[1, 2] == 0)
        ), "ERROR during writing lmp file: cell is not an upper triangular matrix!"

        fp.write("%16.10f %16.10f %s %s\n" % (0, cell[0, 0], "xlo", "xhi"))
        fp.write("%16.10f %16.10f %s %s\n" % (0, cell[1, 1], "ylo", "yhi"))
        fp.write("%16.10f %16.10f %s %s\n" % (0, cell[2, 2], "zlo", "zhi"))
        fp.write(
            "%16.10f %16.10f %16.10f %s %s %s\n"
            % (cell[1, 0], cell[2, 0], cell[2, 1], "xy", "xz", "yz")
        )

        fp.write("Masses\n\n")

        for uid, uel in enumerate(uelems):
            fp.write("%12d %14.8f # %s\n" % (uid + 1, umasses[uid], uel))

        fp.write("\nAtoms # atomic\n\n")
        for cid, coor in enumerate(coors):
            if atom_types is not None:
                typeid = np.where(atom_types[cid] == uelems)[0] + 1
            else:
                typeid = np.where(elems[cid] == uelems)[0] + 1
            if charges is not None:
                if bond_num is not None:
                    # in this case it is a hybrid atom style with a different syntax. Here, it is assumed that charge is first, then bond
                    fp.write(
                        "%12d %5d %16.10f %16.10f %16.10f %16.10f %5d\n"
                        % (cid + 1, typeid, coor[0], coor[1], coor[2], charges[cid], 1)
                    )
                else:
                    fp.write(
                        "%12d %5d %16.10f %16.10f %16.10f %16.10f\n"
                        % (cid + 1, typeid, charges[cid], coor[0], coor[1], coor[2])
                    )
            elif bond_num is not None:
                # style bond - need to specify molecule id
                fp.write(
                    "%12d %5d %5d %16.10f %16.10f %16.10f\n"
                    % (cid + 1, 1, typeid, coor[0], coor[1], coor[2])
                )
            else:
                fp.write(
                    "%12d %5d %16.10f %16.10f %16.10f\n"
                    % (cid + 1, typeid, coor[0], coor[1], coor[2])
                )
