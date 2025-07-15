#!/usr/bin/env python


import argparse
import sys

import ase.io
from ase.filters import UnitCellFilter
from ase.optimize import LBFGS
from mace.calculators import MACECalculator
import numpy as np


def relax_mace(
    model_file,
    structure_file="POSCAR",
    device="cuda",
    box=False,
    max_steps=None,
    write_file=True,
    precision=0.00001,
    box_mask=None,
):

    atoms = ase.io.read(structure_file)
    try:
        calc = MACECalculator(model_file, device=device, enable_cueq=True)
    except:
        calc = MACECalculator(model_file, device=device)
    atoms.calc = calc

    if max_steps is None:
        max_steps = 1000

    if box:
        if box_mask is not None:

            if isinstance(box_mask[0], str):
                box_mask = [x.lower() == "true" or x.lower() == "t" for x in box_mask]

            if len(box_mask) == 3:
                box_mask.extend([0, 0, 0])

            box_mask = np.array(box_mask)

        ucf = UnitCellFilter(atoms, mask=box_mask)
        actual_atoms = ucf
    else:
        actual_atoms = atoms
    optimizer = LBFGS(actual_atoms, trajectory="relax.traj")
    converged = optimizer.run(fmax=precision, steps=max_steps)
    if write_file:
        ase.io.write(f"{structure_file}_relaxed", atoms, format="vasp")
    return atoms, converged


def main():
    parser = argparse.ArgumentParser(description="relax a unit cell with a MACE model")

    parser.add_argument(
        "--struct",
        dest="structure_file",
        type=str,
        default="POSCAR",
        help="input file to relax",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--box",
        dest="box",
        action="store_true",
        default=False,
        help="performs box relaxation in addition to coordinates",
    )
    parser.add_argument(
        "--precision",
        dest="precision",
        type=float,
        default=0.00001,
        help="desired precision for the forces",
    )
    parser.add_argument(
        "--box_mask",
        dest="box_mask",
        type=str,
        default=None,
        nargs="*",
        help="mask to indicate which elements of the strain tensor should be relaxed (use True/False or T/F)",
    )
    parser.add_argument(
        "model_file",
        default="InAs_swa.model",
        nargs="*",
        help="file name for the model parameters",
    )

    args = parser.parse_args()

    afp = open("p_cmnd.log", "a")
    afp.write(" ".join(sys.argv) + "\n")
    afp.close()

    relax_mace(**vars(args))


if __name__ == "__main__":
    main()
