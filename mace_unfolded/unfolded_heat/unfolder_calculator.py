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

import ase.io
import functorch
import torch
from ase import units
from mace import data
from mace.calculators.mace import get_model_dtype
from mace.tools import torch_geometric, torch_tools, utils
from torch.autograd import grad

from mace_unfolded import keys

# this is one unnecessary dependency, but it is nice to evaluate computation times of different stages
from mace_unfolded.unfolded_heat.comms import comms

from .unfolder import Unfolder


class UnfoldedHeatFluxCalculator:

    def __init__(
        self,
        model,
        device=None,
        skin_unfolder=0.1,
        verbose=False,
        report_update=False,
        never_update=False,
        forward=True,
        dtype=None,
        pbc=[True, True, True],
        debug=False,
    ):
        """
        Parameters

            model : mace model to use
            forward : use forward mode autodiff to compute the potential term (this is generally faster)
        """
        self.verbose = verbose
        self.debug = debug
        self.pbc = pbc
        self.num_dim = sum(pbc)
        self.reporter = comms.reporter(silent=(not self.verbose))

        self.forward = forward

        # keep it simple for now - assume model was properly loaded
        self.model = model
        # self.model.training = False  # avoid building graph for second-order derivatives
        # self.model.return_pairwise = False  # don't need this
        # disable_property(self.model.model, "derivative")  # we don't need forces

        # self.device, _ = guess_device_settings(device=device)
        self.device = device
        if self.verbose:
            comms.talk(f"UnfoldedHeatFluxCalculator will use device {self.device}.")

        # mace model works in this way thankfully
        self.model.to(self.device)

        # set the torch default data type
        if dtype is None or dtype == "":
            if dtype == "float64":
                self.dtype = torch.float64
            elif dtype == "float32":
                self.dtype = torch.float32
            model_dtype = get_model_dtype(model)
            self.dtype = model_dtype
        else:
            if dtype == "float64":
                self.model.double()
                self.dtype = torch.float64
            elif dtype == "float32":
                self.model.float()
                self.dtype = torch.float32
        torch_tools.set_default_dtype(dtype)
        self.z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

        # effective cutoff
        cutoff = float(model.get_buffer("r_max").cpu())
        self.r_max = cutoff
        effective_cutoff = cutoff * float(model.get_buffer("num_interactions").cpu())

        self.unfolder = Unfolder(
            effective_cutoff,
            skin=skin_unfolder,
            verbose=verbose,
            report_update=report_update,
            never_update=never_update,
            pbc=self.pbc,
        )

        comms.talk(f"cutoff: {cutoff}, effective: {effective_cutoff}")


        self.pbc_indices = (
            torch.where(torch.Tensor(self.pbc))[0].to(torch.long).to(self.device)
        )

        self.volume = None

        self.temp = None

    def calculate(
        self,
        atoms: ase.Atoms,
        reconfigure=False,
    ):
        """
        Calculates the heat flux and related quantities for a given set of atoms.

        Args:
            atoms (Atoms): The input atoms object.
            reconfigure (bool, optional): Whether to reinitialize the structure. certain optimizations will be fixed

        Returns:
            dict: A dictionary containing the following keys:
                - heat_flux (numpy.ndarray): The calculated heat flux.
                - heat_flux_force_term (numpy.ndarray): The calculated heat flux force term.
                - heat_flux_potential_term (numpy.ndarray): The calculated heat flux potential term.
                - energies (numpy.ndarray): The calculated energies.
        """
        self.reporter.start("calculating")

        n = len(atoms)

        self.reporter.step("unfolding")
        unfolded = self.unfolder(atoms)
        # print(self.unfolder.unfolding.idx)
        # print(self.unfolder.unfolding.directions)
        # print(self.unfolder.unfolding.replica_offsets)
        ase.io.write("POSCAR_unfolding", unfolded.atoms, format="vasp")
        n_unfolded = len(unfolded.atoms)
        
        # print(unfolded)
        if self.verbose:
            comms.talk(f"n: {n}, n unfolded: {n_unfolded}")
        if self.debug:
            ase.io.write("POSCAR_unfolded", unfolded.atoms, format="vasp")
        if self.volume is None or reconfigure:
            self.volume = atoms.get_volume()

        self.reporter.step("load mace batch")

        #### LOAD MACE DATA
        config = data.config_from_atoms(unfolded.atoms)
        
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
        
        # print(data_loader)
        batch = next(iter(data_loader)).to(self.device)

        # just to obtain proper atomic energies
        # I think this makes no sense. The offset was already considered in the model.
        # node_e0 = self.model.atomic_energies_fn(batch["node_attrs"])

        batch_dict = batch.to_dict()
        #### COMPUTATION

        self.reporter.step("preparation")

        velocities_unfolded = torch.tensor(
            (unfolded.atoms.get_velocities() * units.fs * 1000), dtype=self.dtype
        ).to(self.device)

        # potential_barycenter = torch.sum(r_i.unsqueeze(0) * energies, axis=1)   # this was the original, no idea why the energy has 3 dimensions
        
        unfolded_pos = batch_dict["positions"]
        r_i = unfolded_pos.detach()[:n]
        sigma_potential_term = None
        if self.forward:
            # forward mode is faster but more memory heavy
            # do not use for sigma!
            self.reporter.step("virials (potential term) and forward pass")
            energies = torch.zeros(unfolded_pos.shape[0])

            def wrapper(positions):
                batch_dict["positions"] = positions
                # the graph should be built here, so the regular call to the model to allow the backward pass should be unnecessary
                out = self.model(batch_dict, compute_force=False)
                energies = out[keys.energies]
                potential_barycenter = torch.einsum(
                    "ij,i->j", r_i[:, self.pbc], out[keys.energies][:n]
                )
                return potential_barycenter, energies

            # important for computation of the force term later
            unfolded_pos.requires_grad = True
            _, tmp, energies = functorch.jvp(
                wrapper, (unfolded_pos,), (velocities_unfolded,), has_aux=True
            )
            energies = energies[:n]
            hf_potential_term = tmp.detach()
        else:
            self.reporter.step("prediction unfolded")
            model_results_unfolded = self.model(
                batch_dict, training=True, compute_force=False
            )
            energies = model_results_unfolded[keys.energies][:n]
            self.reporter.step("virials (potential term)")
            potential_barycenter = torch.einsum(
                "ij,i->j", r_i[:, self.pbc_indices], energies
            )
            hf_potential_term = torch.zeros(self.num_dim, device=self.device)
            sigma_potential_term = torch.zeros((n_unfolded, self.num_dim, 3), device=self.device)
            for alpha in range(self.num_dim):
                tmp = (
                    grad(
                        potential_barycenter[alpha],
                        unfolded_pos,  # converted_unfolded.inputs["_positions"],
                        retain_graph=True,
                    )[0]
                    .detach()
                    .squeeze()
                )
                sigma_potential_term[:, alpha] = tmp
                hf_potential_term[alpha] = torch.sum(tmp * velocities_unfolded)

        self.reporter.step("virials (force term)")

        energy = torch.sum(energies)
        gradient = (
            grad([energy], [unfolded_pos], retain_graph=False)[0].detach().squeeze()
        )

        # self.reporter.step("virials (force term) - sanity check")
        # gradient = (
        #     grad([energies[0]], [unfolded_pos], retain_graph=False)[0].detach().squeeze()
        # )

        # this does not appear to be the same thing as the regular forces
        # I guess the total energy here is different as it also explicitly includes farther away atoms.
        # So the unfolded_pos[:n] are the same, the atomic energies[:n] are the same but energy not equal energy
        # and since this is not a local potential the edges cannot be uniquely assigned to an atomic energy
        # forces = -gradient[:n].detach().cpu().numpy()

        inner = torch.sum(gradient * velocities_unfolded, dim=1)
        # inner = (gradient * velocities_unfolded).sum(axis=1)
        # inner.unsqueeze(1)
        hf_force_term = torch.sum(
            unfolded_pos[:, self.pbc_indices] * inner.unsqueeze(1), dim=0
        ).detach()
        heat_flux = (hf_potential_term - hf_force_term) / self.volume
        sigma_force_term = None
        sigma_full_term = None
        if sigma_potential_term is not None:
            sigma_force_term = torch.einsum(
                "ij,ik->ijk", unfolded_pos[:, self.pbc_indices], gradient
            )
            sigma_full_term = sigma_potential_term - sigma_force_term
            hf_from_sigma = torch.einsum("ijk,ik->j", sigma_full_term, velocities_unfolded) / self.volume
            assert torch.allclose(heat_flux, hf_from_sigma, atol=1e-4), f"ERROR: heat flux from sigma is not equal to heat flux from forces {hf_from_sigma} != {heat_flux}"
        # self.reporter.step("test")
        # hf_force_term = torch.Tensor([0] * self.num_dim).to(self.device)
        # jkl = torch.index_select(unfolded_pos, dim=1, index=torch.Tensor([1,2]).to(self.device))
        # jkl = torch.transpose(unfolded_pos, 0, 1)[test_tens]
        # asdf = torch.sum(unfolded_pos[:, self.pbc_indices] * inner.unsqueeze(1), dim=0).detach().cpu()
        # hf_force_term = asdf.detach().cpu().numpy()
        self.reporter.step("finalise")
        

        self.reporter.done()

        self.results = {
            "heat_flux": heat_flux,
            "heat_flux_force_term": hf_force_term,
            "heat_flux_potential_term": hf_potential_term,
            "energies": (energies).detach().cpu().numpy(),
            "sigma": sigma_full_term.detach().cpu().numpy(),    
            "sigma_potential_term": sigma_potential_term.detach().cpu().numpy(),
            "sigma_force_term": sigma_force_term.detach().cpu().numpy(),
            # this following return value is consistent with the output from the ASE calculator - why 2 times node_e0? No idea
            # The offset was already considered in the model. Subtracting it here effectively means that the energy is the interaction energy plus a positive offset. It seems so strange. I would guess there is an error in the MACE ASE calculator.
            # "heat_flux_convective_term": hf_convective_term,
            # "energies": (energies - 2 * node_e0[:n])
            # .detach()
            # .cpu()
            # .numpy(),
        }

        return self.results
