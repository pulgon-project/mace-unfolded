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

from ase.calculators.calculator import Calculator, all_changes
import mace_unfolded.md_tools.lammps_interface as li
from ase.atoms import Atoms
import numpy as np


class LAMMPS_MACE_Calculator(Calculator):
    """
    Class to use the LAMMPS interface like an ASE calculator
    Stole most of the code from my previous MTP lammps calculator - no guarantee everything is working
    """

    def __init__(self, model_file_name: str, init_struct: Atoms, pbc=None, **kwargs) -> None:
        Calculator.__init__(self, **kwargs)

        if pbc is None:
            pbc = init_struct.pbc
        # np.unique sorts the results, we don't want that
        unique_symbols, uinds = np.unique(init_struct.get_chemical_symbols(), return_index=True)
        self.lmps = li.init_lammps_mace(
            init_struct, model_file_name, symbols=np.array(init_struct.get_chemical_symbols())[uinds], pbc=pbc
        )
        self.results = {}
        self._post_setup()
        self.implemented_properties = [
            "energy",
            "forces",
        ]
        self.lmps.cmd.echo("none")
        self.lmps.cmd.log("none")

    def calculate(
        self, atoms: Atoms = None, properties=None, system_changes=all_changes
    ):

        # all changes: list of possible values:
        """
        all_changes = [
            'positions',
            'numbers',
            'cell',
            'pbc',
            'initial_charges',
            'initial_magmoms',
        ]
        if numbers change, we will assume that the number of atoms changed
        we will ignore initial_charges and initial_magmoms
        """

        Calculator.calculate(self, atoms)
        
        # the first evaluation should be fine
        if "numbers" in system_changes and "energy" in self.results.keys():
            raise NotImplementedError(
                "Changing the number of atoms or associated species is not implemented yet"
            )
        if "cell" in system_changes:
            self.update_cell(atoms.get_cell().array)
        if "positions" in system_changes:
            self.update_coors(atoms.positions)

        self.results = {}

        # this sets energy and forces
        self.calc_energy()

    def update_coors(self, coors):
        """
        update the coordinates internally in the lammps object if the number of atoms did not change
        """
        self.lmps.scatter_atoms("x", 1, 3, np.ctypeslib.as_ctypes(coors))
        return

    def update_cell(self, cell):
        """
        update the size of the simulation box and rescale the atoms
        """
        comm = (
            "change_box all x final %16.10f %16.10f y final %16.10f %16.10f z final %16.10f %16.10f xy final %16.10f xz final %16.10f yz final %16.10f %s"
            % (
                0.0,
                cell[0, 0],
                0.0,
                cell[1, 1],
                0.0,
                cell[2, 2],
                cell[1, 0],
                cell[2, 0],
                cell[2, 1],
                "remap units box",
            )
        )
        self.lmps.command(comm)
        return

    def calc_forces(self):
        """
        get the forces from the current structure and return a numpy array
        """
        self.lmps.command("run 0 pre yes post no")

        return self._get_forces()

    def calc_energy(self):
        """
        calculates and returns the total energy of a single point
        """
        self.calc_forces()
        return self._get_energy()

    def _get_forces(self):
        """
        get the forces from the current structure and return a numpy array
        """

        self.results["forces"] = np.ctypeslib.as_array(
            self.lmps.gather_atoms("f", 1, 3)
        )
        self.results["forces"].reshape((-1, 3))
        return self.results["forces"]

    def _get_energy(self):
        """
        returns the total energy of a single point
        """
        self.results["energy"] = self.lmps.get_thermo("etotal")
        return self.results["energy"]
    


    def get_cell(self):
        """
        return the current cell in matrix form lammps
        TODO: double check convention
        """
        var = [
            "boxxlo",
            "boxxhi",
            "boxylo",
            "boxyhi",
            "boxzlo",
            "boxzhi",
            "xy",
            "xz",
            "yz",
        ]
        cell_raw = {}
        for v in var:
            cell_raw[v] = self.lmps.extract_global(v)
        # currently only orthorombic
        cell = np.zeros([3, 3], "d")
        cell[0, 0] = cell_raw["boxxhi"] - cell_raw["boxxlo"]
        cell[1, 1] = cell_raw["boxyhi"] - cell_raw["boxylo"]
        cell[2, 2] = cell_raw["boxzhi"] - cell_raw["boxzlo"]
        # set tilting
        cell[1, 0] = cell_raw["xy"]
        cell[2, 0] = cell_raw["xz"]
        cell[2, 1] = cell_raw["yz"]
        return cell

    def get_stress_tensor(self):
        """
        get the current stress tensor (in metal units - bar)
        """
        ptensor_flat = np.zeros([6])
        for i, p in enumerate(self.pressure_elements):
            ptensor_flat[i] = self.lmps.extract_variable(p, None, 0)
        ptensor = np.zeros([3, 3], "d")
        # "pxx", "pyy", "pzz", "pxy", "pxz", "pyz"
        ptensor[0, 0] = ptensor_flat[0]
        ptensor[1, 1] = ptensor_flat[1]
        ptensor[2, 2] = ptensor_flat[2]
        ptensor[0, 1] = ptensor_flat[3]
        ptensor[1, 0] = ptensor_flat[3]
        ptensor[0, 2] = ptensor_flat[4]
        ptensor[2, 0] = ptensor_flat[4]
        ptensor[1, 2] = ptensor_flat[5]
        ptensor[2, 1] = ptensor_flat[5]
        return ptensor

    def _post_setup(self,):
        """
        method to setup pressure and other things that are required
        """
        self.pressure_elements = ["pxx", "pyy", "pzz", "pxy", "pxz", "pyz"]
        for p in self.pressure_elements:
            self.lmps.command("variable %s equal %s" % (p, p))

        self.lmps.command("thermo_style custom step temp etotal pe ke pxx pyy pzz")

    