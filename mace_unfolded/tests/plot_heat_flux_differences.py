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

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.backends.backend_pdf

from swieser_scripts.utils.plot_tools import plot_parity

flux_file1 = "example/flux_files_full_local_2/heat_flux.dat"
flux_file2 = "example/flux_files_unf/heat_flux.dat"

data1 = np.genfromtxt(flux_file1, skip_header=1)
data2 = np.genfromtxt(flux_file2, skip_header=1)

xyz_mapper = ["x", "y", "z"]

f1 = plt.figure(1)
for ind in range(3):
    i = ind + 1
    plot_parity(
        data1[:, i],
        data2[:, i],
        xlabel="flux full semilocal",
        ylabel="flux unfolded",
        figname="example/parity.png",
    )
plt.tight_layout()

f2, axs = plt.subplots(3, 1)
axs = axs.ravel()
for ind in range(3):
    i = ind + 1
    plt.sca(axs[ind])
    plt.title(xyz_mapper[ind])
    plt.plot(range(len(data1[:, i])), data1[:, i], label="flux full semilocal", lw=3)
    plt.plot(
        range(len(data2[:, i])), data2[:, i], linestyle=":", label="flux unfolded", lw=3
    )
    plt.legend()
    if ind == 1:
        plt.ylabel("heat flux")
    if ind == 2:
        plt.xlabel("time step")
    plt.tight_layout()


with matplotlib.backends.backend_pdf.PdfPages(
    "example/flux_test_comparison.pdf"
) as pdf:
    pdf.savefig(f1)
    pdf.savefig(f2)
