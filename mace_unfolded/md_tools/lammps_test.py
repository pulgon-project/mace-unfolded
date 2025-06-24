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

import mace_unfolded.md_tools.lammps_interface as li

if __name__ == "__main__":
    model_path = "/share/theochem/sandro.wieser/models/foundation_models/MACE_MP0/models/MACE-OMAT-0/mace-omat-0-medium.model-mliap_lammps_v100.pt"
    struct_path = (
        "/share/theochem/sandro.wieser/test/speed_test_AIP/test_lmp_python/poscar.lmp"
    )
    lmp = li.init_lammps_mace(
        struct_path, model_path, symbols=["As", "In"], pbc=[False, False, True]
    )

    lmp.cmd.velocity(
        "all",
        "create",
        300,
        123,
        "dist",
        "gaussian",
        "rot",
        "yes",
        "mom",
        "yes",
        "sum",
        "yes",
    )
    lmp.cmd.fix("1", "all", "nve/kk")
    lmp.cmd.run(100)

    lmp.close()

