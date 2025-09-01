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

from mace_unfolded.unfolded_heat.comms import comms
import ase.io
from matscipy.neighbours import neighbour_list
import numpy as np
import torch
from mace import data
from mace.tools import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    to_one_hot,
    torch_geometric,
    voigt_to_matrix,
    utils,
    torch_tools,
)
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
import torch.utils.data

NUM_TESTS = 1


def atomic_data_call(
    edge_index,
    positions,
    shifts,
    unit_shifts,
    cell,
    one_hot,
    weight,
    head,
    energy_weight,
    forces_weight,
    stress_weight,
    virials_weight,
    dipole_weight,
    charges_weight,
    forces,
    energy,
    stress,
    virials,
    dipole,
    charges,
):
    cls = data.AtomicData(
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        positions=torch.tensor(positions, dtype=torch.get_default_dtype()),
        shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
        unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
        cell=cell,
        node_attrs=one_hot,
        weight=weight,
        head=head,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        virials_weight=virials_weight,
        dipole_weight=dipole_weight,
        charges_weight=charges_weight,
        forces=forces,
        energy=energy,
        stress=stress,
        virials=virials,
        dipole=dipole,
        charges=charges,
    )
    return cls

def atomic_data_call_custom(
    edge_index,
    positions,
    shifts,
    unit_shifts,
    cell,
    node_attrs,   # one_hot
    weight,
    head,
    energy_weight,
    forces_weight,
    stress_weight,
    virials_weight,
    dipole_weight,
    charges_weight,
    forces,
    energy,
    stress,
    virials,
    dipole,
    charges,
):
    # changing this will most likely be complicated
    
    num_nodes = node_attrs.shape[0]

    assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
    assert positions.shape == (num_nodes, 3)
    assert shifts.shape[1] == 3
    assert unit_shifts.shape[1] == 3
    assert len(node_attrs.shape) == 2
    assert weight is None or len(weight.shape) == 0
    assert head is None or len(head.shape) == 0
    assert energy_weight is None or len(energy_weight.shape) == 0
    assert forces_weight is None or len(forces_weight.shape) == 0
    assert stress_weight is None or len(stress_weight.shape) == 0
    assert virials_weight is None or len(virials_weight.shape) == 0
    assert dipole_weight is None or dipole_weight.shape == (1, 3), dipole_weight
    assert charges_weight is None or len(charges_weight.shape) == 0
    assert cell is None or cell.shape == (3, 3)
    assert forces is None or forces.shape == (num_nodes, 3)
    assert energy is None or len(energy.shape) == 0
    assert stress is None or stress.shape == (1, 3, 3)
    assert virials is None or virials.shape == (1, 3, 3)
    assert dipole is None or dipole.shape[-1] == 3
    assert charges is None or charges.shape == (num_nodes,)
    # Aggregate data
    data = {
        "num_nodes": num_nodes,
        "edge_index": edge_index,
        "positions": positions,
        "shifts": shifts,
        "unit_shifts": unit_shifts,
        "cell": cell,
        "node_attrs": node_attrs,
        "weight": weight,
        "head": head,
        "energy_weight": energy_weight,
        "forces_weight": forces_weight,
        "stress_weight": stress_weight,
        "virials_weight": virials_weight,
        "dipole_weight": dipole_weight,
        "charges_weight": charges_weight,
        "forces": forces,
        "energy": energy,
        "stress": stress,
        "virials": virials,
        "dipole": dipole,
        "charges": charges,
    }
    
    #super().__init__(**data)

    self.x = x
    self.edge_index = edge_index
    self.edge_attr = edge_attr
    self.y = y
    self.pos = pos
    self.normal = normal
    self.face = face
    for key, item in kwargs.items():
        if key == "num_nodes":
            self.__num_nodes__ = item
        else:
            self[key] = item

    if edge_index is not None and edge_index.dtype != torch.long:
        raise ValueError(
            (
                f"Argument `edge_index` needs to be of type `torch.long` but "
                f"found type `{edge_index.dtype}`."
            )
        )

    if face is not None and face.dtype != torch.long:
        raise ValueError(
            (
                f"Argument `face` needs to be of type `torch.long` but found "
                f"type `{face.dtype}`."
            )
        )
    follow_batch = None
    exclude_keys = None
    dataloader = torch.utils.data.DataLoader(dataset,
            1,
            False,
            collate_fn=Collater(follow_batch, exclude_keys),
            drop_last=False,)

    return dataloader


def custom_batch_test(data, species, num_species, natoms, nghosts, dtype="float64", device="cuda"):
    # this should create a working batch dict just with a lammps data object - the difficulty will just be obtaining said lammps data object
    batch_dict = {
            "vectors": torch.as_tensor(data.rij).to(dtype).to(device),
            "node_attrs": torch.nn.functional.one_hot(
                species.to(device), num_classes=num_species
            ).to(dtype),
            "edge_index": torch.stack(
                [
                    torch.as_tensor(data.pair_j, dtype=torch.int64).to(device),
                    torch.as_tensor(data.pair_i, dtype=torch.int64).to(device),
                ],
                dim=0,
            ),
            "batch": torch.zeros(natoms, dtype=torch.int64, device=device),
            "lammps_class": data,
            "natoms": (natoms, nghosts),
        }
    return batch_dict

def load_lammps():
    pass

def main():

    model = torch.load(
        f="/share/theochem/sandro.wieser/models/foundation_models/MACE_MP0/models/MACE-OMAT-0/mace-omat-0-medium.model",
        map_location="cuda",
        weights_only=False,
    )

    torch_tools.set_default_dtype("float64")

    reporter = comms.reporter(silent=False)
    reporter.start("evaluation")

    pbc = [False, False, False]
    cutoff = 6.0

    reporter.step("ase read")
    for i in range(NUM_TESTS):
        atoms = ase.io.read("POSCAR_unfolding")

    reporter.step("build config")
    for i in range(NUM_TESTS):
        z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        keyspec = data.KeySpecification(info_keys={}, arrays_keys={"charges": False})
        config = data.config_from_atoms(
            atoms, key_specification=keyspec, head_name="Default"
        )

    reporter.step("neighlist matscipy")
    true_self_interaction = False
    for i in range(NUM_TESTS):
        cell = atoms.cell.array
        positions = atoms.positions
        sender, receiver, unit_shifts = neighbour_list(
            quantities="ijS",
            pbc=pbc,
            cell=cell,
            positions=positions,
            cutoff=cutoff,
        )

    reporter.step("remainder neighlist")
    for i in range(NUM_TESTS):
        if not true_self_interaction:
            # Eliminate self-edges that don't cross periodic boundaries
            true_self_edge = sender == receiver
            true_self_edge &= np.all(unit_shifts == 0, axis=1)
            keep_edge = ~true_self_edge

            # Note: after eliminating self-edges, it can be that no edges remain in this system
            sender = sender[keep_edge]
            receiver = receiver[keep_edge]
            unit_shifts = unit_shifts[keep_edge]

        # Build output
        edge_index = np.stack((sender, receiver))  # [2, n_edges]

        # From the docs: With the shift vector S, the distances D between atoms can be computed from
        # D = positions[j]-positions[i]+S.dot(cell)
        shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    reporter.step("one hot")
    for i in range(NUM_TESTS):
        heads = ["Default"]
        indices = atomic_numbers_to_indices(config.atomic_numbers, z_table=z_table)
        one_hot = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )
        try:
            head = torch.tensor(heads.index(config.head), dtype=torch.long)
        except ValueError:
            head = torch.tensor(len(heads) - 1, dtype=torch.long)

    reporter.step("transforming arrays")
    for i in range(NUM_TESTS):
        cell = (
            torch.tensor(cell, dtype=torch.get_default_dtype())
            if cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )

        num_atoms = len(config.atomic_numbers)

        weight = (
            torch.tensor(config.weight, dtype=torch.get_default_dtype())
            if config.weight is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        energy_weight = (
            torch.tensor(
                config.property_weights.get("energy"), dtype=torch.get_default_dtype()
            )
            if config.property_weights.get("energy") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        forces_weight = (
            torch.tensor(
                config.property_weights.get("forces"), dtype=torch.get_default_dtype()
            )
            if config.property_weights.get("forces") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        stress_weight = (
            torch.tensor(
                config.property_weights.get("stress"), dtype=torch.get_default_dtype()
            )
            if config.property_weights.get("stress") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        virials_weight = (
            torch.tensor(
                config.property_weights.get("virials"), dtype=torch.get_default_dtype()
            )
            if config.property_weights.get("virials") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        dipole_weight = (
            torch.tensor(
                config.property_weights.get("dipole"), dtype=torch.get_default_dtype()
            )
            if config.property_weights.get("dipole") is not None
            else torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.get_default_dtype())
        )
        if len(dipole_weight.shape) == 0:
            dipole_weight = dipole_weight * torch.tensor(
                [[1.0, 1.0, 1.0]], dtype=torch.get_default_dtype()
            )
        elif len(dipole_weight.shape) == 1:
            dipole_weight = dipole_weight.unsqueeze(0)

        charges_weight = (
            torch.tensor(
                config.property_weights.get("charges"), dtype=torch.get_default_dtype()
            )
            if config.property_weights.get("charges") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        forces = (
            torch.tensor(
                config.properties.get("forces"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("forces") is not None
            else torch.zeros(num_atoms, 3, dtype=torch.get_default_dtype())
        )
        energy = (
            torch.tensor(
                config.properties.get("energy"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("energy") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )
        stress = (
            voigt_to_matrix(
                torch.tensor(
                    config.properties.get("stress"), dtype=torch.get_default_dtype()
                )
            ).unsqueeze(0)
            if config.properties.get("stress") is not None
            else torch.zeros(1, 3, 3, dtype=torch.get_default_dtype())
        )
        virials = (
            voigt_to_matrix(
                torch.tensor(
                    config.properties.get("virials"), dtype=torch.get_default_dtype()
                )
            ).unsqueeze(0)
            if config.properties.get("virials") is not None
            else torch.zeros(1, 3, 3, dtype=torch.get_default_dtype())
        )
        dipole = (
            torch.tensor(
                config.properties.get("dipole"), dtype=torch.get_default_dtype()
            ).unsqueeze(0)
            if config.properties.get("dipole") is not None
            else torch.zeros(1, 3, dtype=torch.get_default_dtype())
        )
        charges = (
            torch.tensor(
                config.properties.get("charges"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("charges") is not None
            else torch.zeros(num_atoms, dtype=torch.get_default_dtype())
        )

    reporter.step("creating data object")
    for i in range(NUM_TESTS):
        cls = atomic_data_call(
            edge_index,
            config.positions,
            shifts,
            unit_shifts,
            cell,
            one_hot,
            weight,
            head,
            energy_weight,
            forces_weight,
            stress_weight,
            virials_weight,
            dipole_weight,
            charges_weight,
            forces,
            energy,
            stress,
            virials,
            dipole,
            charges,
        )

    reporter.step("calling data loader")
    for i in range(NUM_TESTS):
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[cls],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

    reporter.step("to GPU")
    for i in range(NUM_TESTS):
        # batch = next(iter(data_loader)).to("cpu")
        batch = next(iter(data_loader)).to("cuda")
    reporter.step("to dict")
    for i in range(NUM_TESTS):
        batch_clone = batch.clone()
        batch_clone["node_attrs"].requires_grad_(True)
        batch_clone["positions"].requires_grad_(True)
        batch_dict = batch_clone.to_dict()

    reporter.done()

    model.to("cuda")
    for param in model.parameters():
        param.requires_grad = False
    out = model(
        batch_dict, training=False, compute_stress=False, compute_edge_forces=False
    )


if __name__ == "__main__":
    for j in range(10):
        main()
