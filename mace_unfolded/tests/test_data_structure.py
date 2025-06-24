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
from mace import data
from mace.tools import torch_geometric, utils, torch_tools
from mace.calculators.mace import get_model_dtype
import ase.io

model_file = "/home/swieser/cellar/InAs/wurtzite/training/PBEsol/mace_400/mace_torch_rmax_5/InAs_swa.model"
# 2x2x2 
# struct_file_conv = "/home/swieser/cellar/InAs/wurtzite/training/PBEsol/mace_400/mace_torch_rmax_5/zb/POSCAR"
struct_file_222 = "/home/swieser/cellar/InAs/wurtzite/training/PBEsol/mace_400/mace_torch_rmax_5/222/SPOSCAR"
device = "cuda"

atoms = ase.io.read(struct_file_222)


model = torch.load(model_file, map_location=device)
model.to(device)

model_dtype = get_model_dtype(model)
print("dtype:", model_dtype)
default_dtype = model_dtype
torch_tools.set_default_dtype(default_dtype)

z_table = utils.AtomicNumberTable(
            [int(z) for z in model.atomic_numbers]
        )

r_max = float(model.get_buffer("r_max").cpu())

config = data.config_from_atoms(atoms, charges_key=False)
print(config)
data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=z_table, cutoff=r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
print(data_loader)
batch = next(iter(data_loader)).to(device)
# hmmm, in the mace calculator this is subtracted from the node energies. However, the sum seems to be consistent with the total energy if you don't substract it.
node_e0 = model.atomic_energies_fn(batch["node_attrs"])
print("node_e0",node_e0)
# print(batch.to_dict())
output = model(batch.to_dict(), training=False)
print("output")
for key in output:
    print(key)
    print(output[key])

n = len(atoms)

print(n)

e_i = output["node_energy"][ :n]
print(e_i)
print(torch.sum(e_i))
print(output["energy"])

# a = torch.from_numpy(atoms.get_positions()).to(device)
# print(a)
# print(e_i)
# tsum = torch.einsum("ij,i->j",a,e_i)
# print(tsum)
# hn = int(n/2)
# print(torch.sum(a[:hn],axis=0))
# print(torch.sum(a[hn:],axis=0))
# # torch.sum(a.unsqueeze(0)*e_i,axis=0)

# print(batch.to_dict().keys())
# print(batch.to_dict()["positions"])