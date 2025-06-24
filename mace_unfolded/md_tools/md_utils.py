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

import time

import ase
import numpy as np
from ase.atoms import Atoms
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


def fix_angular_momenta(atoms: Atoms, rescale=False):
    """
    Set the angular velocity to zero.

    Args:
        atoms (ase.Atoms): ASE Atoms object.
        rescale (bool): Whether to rescale the velocities to conserve total energy.

    Returns:
        the ase.Atoms object
    """

    com = atoms.get_center_of_mass()
    dists = atoms.get_positions(wrap=False) - com
    ang_mom = np.cross(dists, atoms.get_momenta()).sum(0)
    inertia, evecs = atoms.get_moments_of_inertia(vectors=True)
    omega = np.dot(np.dot(ang_mom, evecs.T) / inertia, evecs)
    # need to accomplish angular momenta of 0 while conserving the total energy.
    correction = np.cross(omega, dists, axisa=0, axisb=1)
    if rescale:
        old_ekin = atoms.get_kinetic_energy()
    atoms.set_velocities(atoms.get_velocities() - correction)

    if rescale:
        new_ekin = atoms.get_kinetic_energy()
        atoms.set_velocities(atoms.get_velocities() * np.sqrt(old_ekin / new_ekin))
    return atoms


def print_md_info(current_atoms: Atoms, current_integrator, T_limit=10000):
    n_atoms = len(current_atoms)
    e_pot_per_atom = current_atoms.get_potential_energy() / n_atoms
    e_kin_per_atom = current_atoms.get_kinetic_energy() / n_atoms
    kinetic_temperature = e_kin_per_atom / (1.5 * ase.units.kB)
    print(
        f"Step #{current_integrator.nsteps + 1}: "
        f"Epot = {e_pot_per_atom} eV / atom, T = {kinetic_temperature} K, "
        f"E = {e_pot_per_atom + e_kin_per_atom} eV / atom",
        flush=True,
    )
    if kinetic_temperature > T_limit:
        print("unstable")
        exit()


def md_equilibration(
    atoms: Atoms,
    temperature,
    dt=1.0,
    n_equil=1000,
    T_limit=None,
    rng=None,
    trajectory: Trajectory = None,
    traj_output_interval=100,
    init_velocities=True,
    zero_angular_momenta=True,
):
    """
    Equilibrates the given Atoms object by performing a molecular dynamics simulation.

    Args:
        atoms (Atoms): The Atoms object representing the system to be equilibrated.
        temperature (float): The target temperature for the simulation in Kelvin.
        dt (float, optional): The time step in femtoseconds. Defaults to 1.0.
        n_equil (int, optional): The number of time steps to run the simulation. Defaults to 1000.
        T_limit (float, optional): The maximum allowed temperature in Kelvin. If not provided, it is set to 5 times the target temperature. Defaults to None.
        rng (numpy.random.Generator, optional): The random number generator to use for the simulation. If not provided, a new random number generator is created. Defaults to None.
        trajectory (Trajectory, optional): The Trajectory object to write the simulation trajectory to. If provided, the simulation output is written at regular intervals. Defaults to None.
        traj_output_interval (int, optional): The number of time steps between each output in the trajectory. Only used if trajectory is provided. Defaults to 100.
        init_velocities (bool, optional): Whether to initialize the velocities according to the Maxwell-Boltzmann distribution. Defaults to True.
        zero_angular_momenta (bool, optional): Whether to set the center of mass rotation to zero at the end. Defaults to True.

    Returns:
        Atoms: The equilibrated Atoms object.
    """

    if T_limit is None:
        T_limit = temperature * 5

    if rng is None:
        rng = np.random.default_rng(seed=np.random.randint(1, 100000))

    if init_velocities:
        MaxwellBoltzmannDistribution(atoms, temperature_K=float(temperature), rng=rng)

    integrator = Langevin(
        atoms,
        dt * ase.units.fs,
        temperature_K=float(temperature),
        friction=2e-2,
        rng=rng,
    )

    integrator.attach(
        print_md_info,
        interval=1,
        current_atoms=atoms,
        current_integrator=integrator,
        T_limit=T_limit,
    )
    if trajectory is not None:
        integrator.attach(
            trajectory.write,
            interval=traj_output_interval,
            current_integrator=integrator,
        )

    t1 = time.time()
    integrator.run(n_equil)

    if zero_angular_momenta:
        # set center of mass rotation to zero
        atoms = fix_angular_momenta(atoms, rescale=True)

    t_equil = time.time() - t1
    print("equilibration time:", t_equil, flush=True)

    return atoms
