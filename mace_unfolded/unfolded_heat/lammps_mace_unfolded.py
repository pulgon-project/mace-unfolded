# Implementation of a MACE model wrapper to compute the heat flux on-the-fly


import torch
import time
import logging
import os
import sys
from ase.data import chemical_symbols
from e3nn.util.jit import compile_mode
from contextlib import contextmanager

from typing import Dict, Tuple
from mace_unfolded import keys

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:

    class MLIAPUnified:
        def __init__(self):
            pass


class MACELammpsConfig:
    # still unchanged
    """Configuration settings for MACE-LAMMPS integration."""

    def __init__(self):
        self.debug_time = self._get_env_bool("MACE_TIME", False)
        self.debug_profile = self._get_env_bool("MACE_PROFILE", False)
        self.profile_start_step = int(os.environ.get("MACE_PROFILE_START", "5"))
        self.profile_end_step = int(os.environ.get("MACE_PROFILE_END", "10"))
        self.allow_cpu = self._get_env_bool("MACE_ALLOW_CPU", False)
        self.force_cpu = self._get_env_bool("MACE_FORCE_CPU", False)

    @staticmethod
    def _get_env_bool(var_name: str, default: bool) -> bool:
        return os.environ.get(var_name, str(default)).lower() in (
            "true",
            "1",
            "t",
            "yes",
        )


@contextmanager
def timer(name: str, enabled: bool = True):
    """Context manager for timing code blocks."""
    # still unchanged
    if not enabled:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.info(f"Timer - {name}: {elapsed*1000:.3f} ms")


@compile_mode("script")
class MACEEdgeForcesWrapper(torch.nn.Module):
    """Wrapper that adds per-pair force computation to a MACE model."""

    # still unchanged
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)

        if not hasattr(model, "heads"):
            model.heads = ["Default"]

        head_name = kwargs.get("head", model.heads[-1])
        head_idx = model.heads.index(head_name)
        self.register_buffer("head", torch.tensor([head_idx], dtype=torch.long))

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute energies and per-pair forces."""
        data["head"] = self.head

        out = self.model(
            data,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_edge_forces=True,
            lammps_mliap=True,
        )

        node_energy = out["node_energy"]
        # we need to pad the virtual edges with zeros
        if "in_rcut" in data.keys():
            pair_forces = torch.zeros(
                (data["in_rcut"].shape[0], 3),
                device=out["edge_forces"].device,
                dtype=out["edge_forces"].dtype,
            )
            pair_forces[data["in_rcut"]] = out["edge_forces"]
        else:
            pair_forces = out["edge_forces"]
        total_energy = out["energy"][0]

        if pair_forces is None:
            pair_forces = torch.zeros_like(data["vectors"])

        return total_energy, node_energy, pair_forces


class LAMMPS_MLIAP_MACE_HEAT(MLIAPUnified):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.config = MACELammpsConfig()
        self.model = MACEEdgeForcesWrapper(model, **kwargs)
        self.element_types = [chemical_symbols[s] for s in model.atomic_numbers]
        self.num_species = len(self.element_types)
        self.rcutfac = 0.5 * float(model.r_max) * model.num_interactions
        self.rcut = float(model.r_max)
        self.ndescriptors = 1
        self.nparams = 1
        self.dtype = model.r_max.dtype
        self.device = "cpu"
        self.initialized = False
        self.step = 0
        self.hevery = 1
        self.hf_dir = "flux_files_mliap"
        self.hf_skip = 0
        self.set_pbc([True, True, True])  # for computing the heat flux

    def _initialize_device(self, data):
        using_kokkos = "kokkos" in data.__class__.__module__.lower()

        if using_kokkos and not self.config.force_cpu:
            device = torch.as_tensor(data.elems).device
            if device.type == "cpu" and not self.config.allow_cpu:
                raise ValueError(
                    "GPU requested but tensor is on CPU. Set MACE_ALLOW_CPU=true to allow CPU computation."
                )
        else:
            device = torch.device("cpu")

        self.device = device
        self.model = self.model.to(device)
        logging.info(f"MACE model initialized on device: {device}")
        self.initialized = True

    def compute_forces(self, data):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        natoms = data.nlocal
        ntotal = data.ntotal
        nghosts = ntotal - natoms
        npairs = data.npairs
        species = torch.as_tensor(data.elems, dtype=torch.int64)
        # logging.info(
        #     f"natoms: {natoms}, nghosts: {nghosts}, npairs: {npairs}, species: {species} ({species.shape})"
        # )
        # logging.info(f"iatoms: {dir(data)} {type(data)}")
        if not self.initialized:
            self._initialize_device(data)

        self.step += 1
        self._manage_profiling()

        if natoms == 0 or npairs <= 1:
            return

        with timer("total_step", enabled=self.config.debug_time):
            with timer("prepare_batch", enabled=self.config.debug_time):
                batch = self._prepare_batch(data, natoms, nghosts, species)

            with timer("model_forward", enabled=self.config.debug_time):
                _, atom_energies, pair_forces = self.model(batch)

                if self.device.type != "cpu":
                    torch.cuda.synchronize()
            if self.step % self.hevery == 0 and self.step >= self.hf_skip:
                with timer("heat", enabled=self.config.debug_time):
                    self._compute_heat(batch)
            with timer("update_lammps", enabled=self.config.debug_time):
                self._update_lammps_data(data, atom_energies, pair_forces, natoms)

    def _compute_heat(self, batch):
        """
        a few potential issues:
        - do edges between ghost atoms exist? (in which case they probably need to be created)
        - do we have the velocities here? (solution: just compute sigma, however, this makes the on-the-fly aspect more difficult)
        - we have the pair distances, but do we have the positions?
        """

        n_unfolded = batch["natoms"][0] + batch["natoms"][1]
        n = batch["natoms"][0]
        unfolded_pos = batch["positions"]
        velocities_unfolded = batch["velocities"]
        edge_indices = batch["edge_index"]
        # 2 questions need to be answered: are there edges between the ghost atoms? Answer: no. and are positions/velocities correct? if one uses comm_modify vel yes, then yes
        # logging.info(
        #     f"n_unfolded: {n_unfolded}, n: {n}, unfolded_pos: {unfolded_pos.shape}, velocities_unfolded: {velocities_unfolded.shape}"
        # )

        unfolded_pos.requires_grad = True
        ghost_pos = unfolded_pos[n:]
        # Problem 1: the edges are only defined for the actual atoms, no interactions between ghosts, this should not be that hard to remedy considering we don't have to worry about periodic boundary conditions
        ghost_dist = torch.cdist(
            ghost_pos, ghost_pos
        )  # is it faster to just use cdist here in general for all positions?
        mask = (ghost_dist <= self.rcut) & (ghost_dist > 0)
        g_i, g_j = torch.triu(mask, diagonal=1).nonzero(as_tuple=True)
        g_i += n
        g_j += n
        g_edges = torch.stack(
            [
                g_j,
                g_i,
            ],
            dim=0,
        )
        edge_indices = torch.concat([edge_indices, g_edges], dim=1)
        # for i in range(len(g_edges[0])):
        #     logging.info(f"ghost edge_indices: {g_edges[0][i]}, {g_edges[1][i]}")
        # for i in range(len(edge_indices[0])):
        #     logging.info(f"edge_indices: {edge_indices[0][i]}, {edge_indices[1][i]}")

        # Problem 2: we need to compute the r_ij anew from the positions such that they are in the pytorch graph, good thing is, we have the pair indices
        r_ij = unfolded_pos[edge_indices[0]] - unfolded_pos[edge_indices[1]]
        batch["edge_index"] = edge_indices
        batch["vectors"] = r_ij

        # since we are not in sync with the lammps class any more, we need to add the remaining relevant quantities to the batch
        # we need to add a cell, for this we just use a big box because we don't care about the periodic boundary conditions
        big_box = torch.ones(3, dtype=self.dtype, device=self.device) * (
            torch.max(unfolded_pos) - torch.min(unfolded_pos) + 20
        )
        batch["cell"] = torch.diag(big_box)
        batch["head"] = self.model.head
        batch["ptr"] = torch.tensor(
            [0, n_unfolded], dtype=self.dtype, device=self.device
        )
        # unit shifts are meant to be the shift of the periodic image. However, we treat this as non-periodic here, so it should be zero
        batch["unit_shifts"] = torch.zeros(
            (len(edge_indices[0]), 3), dtype=self.dtype, device=self.device
        )
        batch["shifts"] = torch.zeros(
            (len(edge_indices[0]), 3), dtype=self.dtype, device=self.device
        )
        # we make MACE think all atoms are real
        batch["batch"] = torch.zeros(n_unfolded, dtype=torch.int64, device=self.device)
        # batch["lammps_class"] = None
        for key in batch:
            try:
                print(key, batch[key].shape)
            except:
                print(key)

        r_i = unfolded_pos.detach()[:n]
        sigma_potential_term = None
        model_results_unfolded = self.model.model(
            batch,
            training=True,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_edge_forces=False,
            lammps_mliap=False,
        )
        energies = model_results_unfolded[keys.energies][:n]
        potential_barycenter = torch.einsum(
            "ij,i->j", r_i[:, self.pbc_indices], energies
        )
        hf_potential_term = torch.zeros(self.num_dim, device=self.device)
        # sigma_potential_term = torch.zeros(
        #     (n_unfolded, self.num_dim, 3), device=self.device
        # )
        for alpha in range(self.num_dim):
            tmp = (
                torch.autograd.grad(
                    potential_barycenter[alpha],
                    unfolded_pos,  # converted_unfolded.inputs["_positions"],
                    retain_graph=True,
                )[0]
                .detach()
                .squeeze()
            )
            # sigma_potential_term[:, alpha] = tmp
            hf_potential_term[alpha] = torch.sum(tmp * velocities_unfolded)

        energy = torch.sum(energies)
        gradient = (
            torch.autograd.grad([energy], [unfolded_pos], retain_graph=False)[0]
            .detach()
            .squeeze()
        )

        inner = torch.sum(gradient * velocities_unfolded, dim=1)
        hf_force_term = torch.sum(
            unfolded_pos[:, self.pbc_indices] * inner.unsqueeze(1), dim=0
        ).detach()
        heat_flux = hf_potential_term - hf_force_term  # / self.volume
        # sigma_force_term = None
        # sigma_full_term = None
        # if sigma_potential_term is not None:
        #     sigma_force_term = torch.einsum(
        #         "ij,ik->ijk", unfolded_pos[:, self.pbc_indices], gradient
        #     )
        #     sigma_full_term = sigma_potential_term - sigma_force_term
        #     hf_from_sigma = (
        #         torch.einsum("ijk,ik->j", sigma_full_term, velocities_unfolded)
        #         #/ self.volume
        #     )
        #     # we use a lower atol here since we don't divide by the volume
        #     assert torch.allclose(
        #         heat_flux, hf_from_sigma, atol=1e-2
        #     ), f"ERROR: heat flux from sigma is not equal to heat flux from forces {hf_from_sigma} != {heat_flux}"

        velocities = velocities_unfolded[:n]

        # conversion factor for units metal
        eV2J = 1.60218e-19
        J2eV = 1.0 / eV2J
        amu2kg = 1.66054e-27
        A2m = 1.0e-10
        ps2s = 1.0e-12
        kine2J = amu2kg * A2m**2 / ps2s**2
        kB = 1.380649e-23
        # this is one of the reasons why some fix or compute style would be nicer - easier to access to these kind of attributes
        # it would also be an option to add the result numbers to the MLIAP Data object and make it accessible within the lammps script
        # only real masses
        masses = batch["masses"][:n]
        # for i in range(len(masses)):
        #     logging.info(
        #         f"mass {i}: {masses[i].detach().cpu().numpy()} amu -> {masses[i].detach().cpu().numpy() * amu2kg} kg"
        #     )
        kinetic_energies = torch.sum((velocities**2), axis=1) * masses / 2 * kine2J
        logging.info(f"kinetic energies: {kinetic_energies.detach().cpu().numpy()}")
        temperature = torch.mean(kinetic_energies) / kB * 2 / 3
        atomic_energies = energies + kinetic_energies * J2eV
        hf_convective_term = (
            velocities.T @ atomic_energies
        )  # / current_atoms.get_volume()

        results = {
            "heat_flux": heat_flux,
            "heat_flux_force_term": hf_force_term,
            "heat_flux_potential_term": hf_potential_term,
            "heat_flux_convective_term": hf_convective_term,
            "energies": (energies).detach().cpu().numpy(),
            "temperature": temperature.detach().cpu().numpy(),
            # "sigma": sigma_full_term.detach().cpu().numpy(),
            # "sigma_potential_term": sigma_potential_term.detach().cpu().numpy(),
            # "sigma_force_term": sigma_force_term.detach().cpu().numpy(),
        }

        self.write_hf_results(results)

    def write_hf_results(self, results_dict):
        # if this ends up being slow, batch the output
        if not os.path.exists(self.hf_dir):
            os.mkdir(self.hf_dir)
        flux_file_name = os.path.join(self.hf_dir, "heat_flux.dat")
        flux_comp_file_name = os.path.join(self.hf_dir, "heat_flux_components.dat")
        temp = results_dict["temperature"]
        flux = results_dict["heat_flux"]
        pot_term = results_dict["heat_flux_potential_term"]
        force_term = results_dict["heat_flux_force_term"]
        hf_convective_term = results_dict["heat_flux_convective_term"]
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
            for quantity in [
                force_term,
                pot_term,
                flux,
                hf_convective_term[self.pbc_indices],
            ]:
                for ind in range(len(quantity)):
                    wstr += "%18.12f " % quantity[ind]
            wstr += "\n"
            hfp.write(wstr)

    def set_pbc(self, pbc):
        self.pbc = pbc
        self.num_dim = sum(pbc)
        self.pbc_indices = (
            torch.where(torch.Tensor(self.pbc))[0].to(torch.long).to(self.device)
        )

    def _prepare_batch(self, data, natoms, nghosts, species):
        """Prepare the input batch for the MACE model."""

        # logging.info(f"rij shape: {data.rij.shape}")
        rij = torch.as_tensor(data.rij).to(self.dtype).to(self.device)
        pair_j = torch.as_tensor(data.pair_j, dtype=torch.int64).to(self.device)
        pair_i = torch.as_tensor(data.pair_i, dtype=torch.int64).to(self.device)
        dists = torch.linalg.norm(rij, dim=1)
        in_rcut = dists <= self.rcut

        rij = rij[in_rcut]
        pair_j = pair_j[in_rcut]
        pair_i = pair_i[in_rcut]

        # this needs to be added to the LAMMPS C++ code
        positions = torch.as_tensor(data.positions).to(self.dtype).to(self.device)
        velocities = torch.as_tensor(data.velocities).to(self.dtype).to(self.device)

        return {
            "vectors": rij,
            "node_attrs": torch.nn.functional.one_hot(
                species.to(self.device), num_classes=self.num_species
            ).to(self.dtype),
            "edge_index": torch.stack(
                [
                    pair_j,
                    pair_i,
                ],
                dim=0,
            ),
            "batch": torch.zeros(natoms, dtype=torch.int64, device=self.device),
            "lammps_class": data,
            "natoms": (natoms, nghosts),
            "in_rcut": in_rcut,
            "positions": positions,
            "velocities": velocities,
            "masses": torch.as_tensor(data.masses).to(self.dtype).to(self.device),
        }

    def _update_lammps_data(self, data, atom_energies, pair_forces, natoms):
        """Update LAMMPS data structures with computed energies and forces."""
        if self.dtype == torch.float32:
            pair_forces = pair_forces.double()
        eatoms = torch.as_tensor(data.eatoms)
        eatoms.copy_(atom_energies[:natoms])
        data.energy = torch.sum(atom_energies[:natoms])
        data.update_pair_forces_gpu(pair_forces)

    def _manage_profiling(self):
        if not self.config.debug_profile:
            return

        if self.step == self.config.profile_start_step:
            logging.info(f"Starting CUDA profiler at step {self.step}")
            torch.cuda.profiler.start()

        if self.step == self.config.profile_end_step:
            logging.info(f"Stopping CUDA profiler at step {self.step}")
            torch.cuda.profiler.stop()
            logging.info("Profiling complete. Exiting.")
            sys.exit()

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass
