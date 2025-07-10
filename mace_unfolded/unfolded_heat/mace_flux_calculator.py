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

import torch

from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from ase import units

from mace.calculators import MACECalculator
from mace import data
from mace.tools import torch_geometric

from mace_unfolded.unfolded_heat.comms import comms


class MACE_flux_calculator(MACECalculator):
    r"""MACE calculator supporting several methods to compute the heat flux"""

    def __init__(
        self, full_semilocal_flux=False, verbose=False, full_local_flux=False, **kwargs
    ):
        """
        Initialize the calculator with optional parameters.

        Parameters:
            full_semilocal_flux (bool): Whether to calculate the full semilocal flux.
            verbose (bool): Whether to enable verbose mode.
            full_local_flux (bool): Whether to calculate the full local flux.
            **kwargs: Additional keyword arguments that will be passed to the MACECalculator constructor.

        Returns:
            None
        """
        super().__init__(**kwargs)

        self.full_semilocal_flux = full_semilocal_flux
        self.full_local_flux = full_local_flux
        self.retain_graph = False
        self.verbose = verbose
        self.reporter = comms.reporter(silent=(not self.verbose))
        if self.full_semilocal_flux:
            self.init_full_semilocal_flux()
        if self.full_local_flux:
            self.init_full_local_flux()

        eV2J = 1.60218e-19
        J2eV = 1.0 / eV2J
        amu2kg = 1.66054e-27
        A2m = 1.0e-10
        ps2s = 1.0e-12
        kine2J = amu2kg * A2m**2 / ps2s**2
        self.kine_conversion2eV = kine2J * J2eV

    def init_full_semilocal_flux(self):
        """
        Initializes the full semilocal flux calculation.

        Parameters:
            None

        Returns:
            None
        """
        self.full_semilocal_flux = True
        self.retain_graph = True

    def init_full_local_flux(self, slow_method=False, exchange_inds=False):
        """
        Initialize full local flux with the option to use a slow method.

        :param slow_method: Boolean indicating whether to use a slow method (default is False)
        :return: None
        """
        self.full_local_flux = True
        self.retain_graph = True
        self.slow_method = slow_method
        self.exchange_inds = exchange_inds

    def set_verbose(self, verbose):
        """
        set verbosity to output computation time breakdowns
        """
        self.verbose = True
        self.reporter = comms.reporter(silent=(not self.verbose))

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        r"""
        Calculate properties.

        Cannot be inhereted fully from the MACECalculator since the graph needs to be retained during the force computation
        """
        self.reporter.start("calculating")
        # call to ASE calculator
        # will not do super here since the MACE calculator does a couple of things we do not want
        # most of the code below is a 1:1 copy
        self.reporter.step("ase super")
        Calculator.calculate(self, atoms)

        self.reporter.step("preparing data")
        # prepare data
        config = data.config_from_atoms(atoms)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            batch = next(iter(data_loader)).to(self.device)
            node_e0 = self.models[0].atomic_energies_fn(batch["node_attrs"])
            compute_stress = True
        else:
            compute_stress = False

        batch_base = next(iter(data_loader)).to(self.device)
        ret_tensors = self._create_result_tensors(
            self.model_type, self.num_models, len(atoms)
        )

        self.results = {}

        # add flux return tensor
        if self.full_semilocal_flux or self.full_local_flux:
            flux_pot = torch.zeros(len(self.models), 3, device=self.device)
            flux_conv = torch.zeros(len(self.models), 3, device=self.device)
            sigma = torch.zeros(len(self.models), len(atoms), 3, 3, device=self.device)
            ret_tensors.update(
                {"flux_pot": flux_pot, "flux_conv": flux_conv, "sigma": sigma}
            )

        self.reporter.step("evaluating models")
        for i, model in enumerate(self.models):
            batch = batch_base.clone()
            batch_dict = batch.to_dict()
            # we need training=True here since we want to retain the graph
            self.reporter.step("base evaluation of graph with forces backwards pass")
            out = model(
                batch_dict, compute_stress=compute_stress, training=self.retain_graph
            )
            if self.model_type in ["MACE", "EnergyDipoleMACE"]:
                ret_tensors["energies"][i] = out["energy"].detach()
                ret_tensors["node_energy"][i] = (out["node_energy"] - node_e0).detach()
                ret_tensors["forces"][i] = out["forces"].detach()
                if out["stress"] is not None:
                    ret_tensors["stress"][i] = out["stress"].detach()
            if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
                ret_tensors["dipole"][i] = out["dipole"].detach()

            if self.full_semilocal_flux:
                self.reporter.step("evaluation of dU_k/dr_ij derivatives")
                # print(batch_dict.keys())
                # print(batch_dict["vectors"].shape)
                # contains a 2xN tensor, where N is the number of edges. dim 0 are the atoms in the interaction.
                # Need to map it back to actual atom indices to compute the heat flux.
                # essentially this entails the innermost sum over the atoms within the cutoff
                # print(batch_dict["edge_index"])
                # the edge indices are R_KJ
                # unique_indices = torch.unique(batch_dict["edge_index"][0])
                natoms = batch_dict["forces"].shape[0]
                # gradient of Ui wrt r_jk, where the sum over k is already performed as in
                # sum ( dU_I/dR_KJ - dU_I/dR_JK )
                gradients = torch.zeros((natoms, natoms, 3), device=self.device)
                forces = torch.zeros((natoms, 3), device=self.device)
                # this loop should compute the full dUk/dr_ij matrix
                for nid, node_energy in enumerate(out["node_energy"]):
                    # Important note: for this to work the main ScaleShiftMACE module needs to be adapted with the line:
                    """
                    data["vectors"] = vectors
                    """
                    # after the call of 'get_edge_vectors_and_lengths'
                    gradient = torch.autograd.grad(
                        node_energy, batch_dict["vectors"], retain_graph=True
                    )[0]

                    # inverted sign since by adressing the first index using batch_dict["edge_index"][0, pid] it returns the index of the connected atom
                    # gradients[nid, batch_dict["edge_index"][0]] -= gradient
                    # gradients[nid, batch_dict["edge_index"][1]] += gradient
                    # # sum over nid and j as we just want to double check the actual forces
                    # forces[batch_dict["edge_index"][0]] += gradient
                    # forces[batch_dict["edge_index"][1]] -= gradient
                    for j in range(batch_dict["edge_index"].shape[1]):
                        gradients[nid, batch_dict["edge_index"][0, j]] -= gradient[j]
                        gradients[nid, batch_dict["edge_index"][1, j]] += gradient[j]
                        forces[batch_dict["edge_index"][0, j]] += gradient[j]
                        forces[batch_dict["edge_index"][1, j]] -= gradient[j]
                self.reporter.step("sums over derivatives to obtain the flux")
                forces_diff = torch.sum(
                    torch.abs(forces - ret_tensors["forces"])
                ) / len(forces)
                assert (
                    forces_diff < 1e-5
                ), f"ERROR: forces from pairwise gradients not consistent! Difference: {forces_diff}"
                # need to get r_ij for entire uc - considering MIC
                r_ij = torch.Tensor(atoms.get_all_distances(mic=True, vector=True)).to(
                    device=self.device
                )
                velocities = torch.Tensor(atoms.get_velocities() * units.fs * 1000).to(
                    device=self.device
                )

                sigma = torch.einsum(
                    "ijkl->jkl", torch.einsum("jik,ijl->ijkl", r_ij, gradients)
                )
                hflux_from_sigma = torch.einsum("ijk,ik->j", sigma, velocities)
                rhand = torch.einsum("ijk,jk->ij", gradients, velocities)
                flux = torch.einsum("jik,ij->k", r_ij, rhand)
                assert torch.allclose(
                    flux, hflux_from_sigma, atol=1e-5
                ), f"ERROR: heat flux from sigma is not equal to direct heat flux {hflux_from_sigma} != {flux}"
                ret_tensors["flux_pot"][i] = flux.detach()
                ret_tensors["sigma"][i] = sigma.detach()

                # there are more efficient implementations, but it will take long anyway
                # jacobian = []
                # for nid, node_energy in enumerate(out["node_energy"]):
                #     dui_drjk = (
                #         torch.autograd.grad(
                #             node_energy,
                #             vectors,
                #         )[0]
                #         .detach()
                #         .squeeze()
                #     )
                #     jacobian.append(dui_drjk)
                # print(jacobian)

            elif self.full_local_flux:
                if not self.slow_method:
                    # only works for local FFs, M=1
                    gradient = torch.autograd.grad(
                        [out["energy"]],
                        [batch_dict["vectors"]],
                        retain_graph=False,
                    )[0]

                    natoms = batch_dict["forces"].shape[0]
                    # edge index is built identically as in MACE
                    partial_forces = torch.zeros(
                        (natoms, natoms, 3), device=self.device
                    )
                    grads = torch.zeros((natoms, natoms, 3), device=self.device)

                    edge_index = batch_dict["edge_index"]
                    # reminder, these might be partial forces, but not the same partial forces as in the slow solution below
                    partial_forces[edge_index[0, :], edge_index[1, :]] -= gradient
                    partial_forces[edge_index[1, :], edge_index[0, :]] += gradient
                    grads[edge_index[0, :], edge_index[1, :]] = gradient
                    # sanity check: the following 2 lines are equivalent to the previous one, was worried about adressing some of the indices multiple times
                    # for gid, grad_i in enumerate(gradient):
                    #     grads[edge_index[0, gid], edge_index[1, gid]] = grad_i
                    grads = grads.negative()
                    forces = partial_forces.sum(dim=0)
                    forces_diff = torch.sum(
                        torch.abs(forces - ret_tensors["forces"])
                    ) / len(forces)
                    assert (
                        forces_diff < 1e-5
                    ), f"ERROR: forces from pairwise gradients not consistent! Difference: {forces_diff}"

                    r_ij = torch.Tensor(
                        atoms.get_all_distances(mic=True, vector=True)
                    ).to(device=self.device)
                    velocities = torch.Tensor(
                        atoms.get_velocities() * units.fs * 1000
                    ).to(device=self.device)
                    if self.exchange_inds:
                        rhand = torch.einsum("ijk,jk->ij", grads, velocities)
                    else:
                        rhand = torch.einsum("jik,jk->ij", grads, velocities)
                    flux = torch.einsum("jik,ij->k", r_ij, rhand)
                    ret_tensors["flux_pot"][i] = flux.detach()

                else:
                    natoms = batch_dict["forces"].shape[0]
                    gradients = torch.zeros((natoms, natoms, 3), device=self.device)

                    for nid, node_energy in enumerate(out["node_energy"]):
                        grad = torch.autograd.grad(
                            node_energy, batch_dict["positions"], retain_graph=True
                        )[0]
                        gradients[nid] = grad
                    # jacrev is very memory heavy
                    # def wrapper(positions):
                    #     batch_dict["positions"] = positions
                    #     # the graph should be built here, so the regular call to the model to allow the backward pass should be unnecessary
                    #     out = model(batch_dict, compute_force=False)
                    #     energies = out["node_energy"]
                    #     return energies
                    # gradients = functorch.jacrev(wrapper)(torch.Tensor(atoms.positions).to(device=self.device))
                    self.reporter.step("sums over derivatives to obtain the flux")
                    partial_forces = gradients.negative()
                    forces = partial_forces.sum(dim=0)

                    forces_diff = torch.sum(
                        torch.abs(forces - ret_tensors["forces"])
                    ) / len(forces)
                    assert (
                        forces_diff < 1e-5
                    ), f"ERROR: forces from pairwise gradients not consistent! Difference: {forces_diff}"
                    # need to get r_ij for entire uc - considering MIC
                    r_ij = torch.Tensor(
                        atoms.get_all_distances(mic=True, vector=True)
                    ).to(device=self.device)
                    velocities = torch.Tensor(
                        atoms.get_velocities() * units.fs * 1000
                    ).to(device=self.device)

                    rhand = torch.einsum("ijk,jk->ij", partial_forces, velocities)
                    flux = torch.einsum("jik,ij->k", r_ij, rhand)
                    ret_tensors["flux_pot"][i] = flux.detach()

            if "flux_pot" in ret_tensors.keys():
                kinetic_energies = (
                    torch.einsum(
                        "i,ij->i",
                        torch.Tensor(atoms.get_masses()).to(device=self.device),
                        velocities**2,
                    )
                    / 2
                ) * self.kine_conversion2eV
                flux_conv = torch.einsum(
                    "i,ij->j", out["node_energy"] + kinetic_energies, velocities
                )
                ret_tensors["flux_conv"][i] = flux_conv.detach()

        self.reporter.step("formatting the output")
        # I doubt the implementation for the heat flux will work for the dipole options but I will leave it in for now
        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            if self.full_semilocal_flux or self.full_local_flux:
                self.results["heat_flux_pot"] = (
                    torch.mean(ret_tensors["flux_pot"], dim=0).cpu().numpy()
                    / atoms.get_volume()
                    * self.energy_units_to_eV
                    / self.length_units_to_A**2
                )
                self.results["heat_flux_conv"] = (
                    torch.mean(ret_tensors["flux_conv"], dim=0).cpu().numpy()
                    / atoms.get_volume()
                    * self.energy_units_to_eV
                    / self.length_units_to_A**2
                )
                self.results["sigma"] = torch.mean(ret_tensors["sigma"], dim=0).cpu().numpy()
            else:
                self.results["heat_flux_pot"] = torch.zeros(3).numpy()
                self.results["heat_flux_conv"] = torch.zeros(3).numpy()

            self.results["energy"] = (
                torch.mean(ret_tensors["energies"], dim=0).cpu().item()
                * self.energy_units_to_eV
            )
            self.results["free_energy"] = self.results["energy"]
            self.results["node_energy"] = (
                torch.mean(ret_tensors["node_energy"] - node_e0, dim=0).cpu().numpy()
            )
            self.results["forces"] = (
                torch.mean(ret_tensors["forces"], dim=0).cpu().numpy()
                * self.energy_units_to_eV
                / self.length_units_to_A
            )
            if self.num_models > 1:
                self.results["energies"] = (
                    ret_tensors["energies"].cpu().numpy() * self.energy_units_to_eV
                )
                self.results["energy_var"] = (
                    torch.var(ret_tensors["energies"], dim=0, unbiased=False)
                    .cpu()
                    .item()
                    * self.energy_units_to_eV
                )
                self.results["forces_comm"] = (
                    ret_tensors["forces"].cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A
                )
            if out["stress"] is not None:
                self.results["stress"] = full_3x3_to_voigt_6_stress(
                    torch.mean(ret_tensors["stress"], dim=0).cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A**3
                )
                if self.num_models > 1:
                    self.results["stress_var"] = full_3x3_to_voigt_6_stress(
                        torch.var(ret_tensors["stress"], dim=0, unbiased=False)
                        .cpu()
                        .numpy()
                        * self.energy_units_to_eV
                        / self.length_units_to_A**3
                    )
        if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
            self.results["dipole"] = (
                torch.mean(ret_tensors["dipole"], dim=0).cpu().numpy()
            )
            if self.num_models > 1:
                self.results["dipole_var"] = (
                    torch.var(ret_tensors["dipole"], dim=0, unbiased=False)
                    .cpu()
                    .numpy()
                )
        self.reporter.done()
