#!/usr/bin/env python

import argparse
import sys

import ase
import ase.io
import phonopy.file_IO
import torch
import torch.func
import numpy as np
import phonopy as ph
from ase import Atoms
from ase.optimize import LBFGS
from phonopy import Phonopy
from phonopy.harmonic.force_constants import full_fc_to_compact_fc
from phonopy.structure.atoms import PhonopyAtoms
from tqdm import tqdm
from mace.calculators import MACECalculator
from mace import data
from mace.tools import torch_geometric
import mace
import time
from packaging.version import Version


def get_site_symmetry(atoms, phonon: Phonopy, symprec=1e-2, round=None):
    # TODO: write more general and put in separate file. Currently only works in z orientation
    from pulgon_tools_wip.detect_generalized_translational_group import (
        CyclicGroupAnalyzer,
    )
    from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer
    from pulgon_tools_wip.utils import (
        Cn,
        brute_force_generate_group,
        find_axis_center_of_nanotube,
        get_independent_atoms,
        get_perms_from_ops,
        get_site_symmetry,
    )
    from pymatgen.core.operations import SymmOp

    if round is None:
        # HACK
        round = -int(np.floor(np.log10(symprec)))
        # print(round)

    obj = LineGroupAnalyzer(atoms, tolerance=symprec)
    nrot = obj.get_rotational_symmetry_number()
    unit_atoms_center = find_axis_center_of_nanotube(atoms)
    cyclic = CyclicGroupAnalyzer(unit_atoms_center, tolerance=symprec)
    try:
        obj_cyclic = CyclicGroupAnalyzer(atoms, tolerance=symprec)
        translation_cyclic_group, monomers = obj_cyclic.get_cyclic_group()
    except:
        translation_cyclic_group = ["T"]

    z_index = phonon.supercell_matrix[2, 2]

    unitcell = PhonopyAtoms(
        symbols=unit_atoms_center.symbols,
        cell=unit_atoms_center.cell,
        scaled_positions=unit_atoms_center.get_scaled_positions(),
    )
    phonon = Phonopy(
        unitcell,
        supercell_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, z_index]],
        primitive_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )

    atoms_sc = Atoms(
        positions=phonon.supercell.positions,
        numbers=phonon.supercell.numbers,
        cell=phonon.supercell.cell,
    )
    # atom_114_center = find_axis_center_of_nanotube(atom_114)
    # set_trace()

    sym = []
    pg1 = obj.get_generators()
    for pg in pg1:
        tmp = SymmOp(pg)
        sym.append(tmp.affine_matrix)

    # this is somewhat hardcoded
    # use DIRECT coordinates for the operations
    if len(translation_cyclic_group) > 1:
        # TODO
        print(
            f"WARNING: cyclic group {translation_cyclic_group} detected. Only supporting ['T'] at the moment"
        )
        # tran1 = []
        # tran1.append(SymmOp.from_rotation_and_translation(Cn(2 * nrot), [0, 0, 1 / 2]))
        # for tran in tran1:
        #     sym.append(tran.affine_matrix)

    # sym = np.round(sym, round)
    # this should be more accourate than the rest
    sym = np.round(sym, 6)

    ops = brute_force_generate_group(sym)
    # print(ops)
    # ops = dimino_affine_matrix(sym)

    ops_sym = []
    for op in ops:
        ops_sym.append(
            SymmOp.from_rotation_and_translation(
                rotation_matrix=op[:3, :3],
                translation_vec=op[:3, 3]
                * cyclic._pure_trans
                * z_index,  # supercell size is 4 in z direction
            )
        )
    perms_ops = get_perms_from_ops(atoms_sc, ops_sym, symprec=symprec, round=round)

    atom_num = get_independent_atoms(perms_ops)
    site_symmetry = get_site_symmetry(atom_num, perms_ops, ops_sym)

    rotations = np.array([op.rotation_matrix for op in ops_sym])

    return site_symmetry, atom_num, perms_ops, rotations


def autodiff_fcs(
    supercell: Atoms,
    supercell_matrix: np.ndarray,
    primitive_matrix: np.ndarray,
    calc: MACECalculator,
    jacrev=False,
    jacfwd=False,
    jac_chunk=None,
    diffable=False,
):
    """
    Calculate the force constants of a system using automatic differentiation.
    Depending on the symmetries of the system this can be faster.

    Also supports computation of the force constants via the torch.func compositions.
    However, this is only recommended if memory is available in abundance.

    Args:
        supercell (ase.Atoms): ASE Atoms object of the supercell.
        supercell_matrix (np.ndarray): Spanned supercell matrix
        primitive_matrix (np.ndarray): Spanned primitive matrix
        calc (MACECalculator): MACECalculator object
        jacrev (bool, optional): Whether to use Jacrev. Defaults to False. Extremely memory hungry.
        jacfwd (bool, optional): Whether to use Jacfwd. Defaults to False. Extremely memory hungry.
        jac_chunk (int, optional): Chunk size for jacrev. Defaults to None.
        diffable (bool, optional): Whether to return differentiable force constants. Defaults to False.

    Returns:
        np.ndarray: The force constant matrix in compact form
    """
    # we need to compute the derivative of all forces in the supercell with respect to the positions
    # in the primitive cell to obtain the force constants in the compact format
    models = calc.models
    charges_key = calc.charges_key
    z_table = calc.z_table
    r_max = calc.r_max
    device = calc.device

    if Version(mace.__version__) <= Version("0.3.12"):
        config = data.config_from_atoms(supercell, charges_key=charges_key)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
    else:
        # for mace versions 0.3.13 and above
        keyspec = data.KeySpecification(
            info_keys={}, arrays_keys={"charges": calc.charges_key}
        )
        config = data.config_from_atoms(
            supercell, key_specification=keyspec, head_name=calc.head
        )
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config,
                    z_table=z_table,
                    cutoff=r_max,
                    heads=calc.available_heads,
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

    batch_base = next(iter(data_loader)).to(device)
    if primitive_matrix is None:
        primitive_matrix = np.diag(np.array([1, 1, 1]))
    multiplicity = int(
        np.round(np.linalg.det(supercell_matrix) / np.linalg.det(primitive_matrix))
    )
    # node_e0 = models[0].atomic_energies_fn(batch["node_attrs"])
    # ret_tensors = calc._create_result_tensors(
    #     calc.model_type, calc.num_models, len(supercell)
    # )
    natoms = len(supercell)
    if diffable:
        gradient = torch.zeros((natoms // multiplicity, natoms, 3, 3), device=device)
    else:
        gradient = np.zeros((natoms // multiplicity, natoms, 3, 3))
    for i, model in enumerate(models):
        batch = batch_base.clone()

        batch["positions"].requires_grad = True

        if jacrev or jacfwd:

            def wrapper(pos: torch.Tensor) -> torch.Tensor:
                batch["positions"] = pos
                # note, this does not work with base mace - you are not allowed to call requires_grad_ for the torch.func compositions
                # it is fine to set requires_grad as a property however
                out = model(batch.to_dict(), compute_stress=False, training=True)
                forces = out["forces"][0::multiplicity]
                return forces

            # primary issue: extremely memory heavy
            t1 = time.time()
            if jacrev:
                a = torch.func.jacrev(wrapper, chunk_size=jac_chunk)(batch["positions"])
            elif jacfwd:
                a = torch.func.jacfwd(wrapper)(batch["positions"])
            print(f"autodiff time: {time.time() - t1}")
            if diffable:
                gradient += torch.transpose(a, 1, 2)
            else:
                gradient += torch.transpose(a, 1, 2).detach().cpu().numpy()
        else:
            # we don't need the stress
            # however, we need to retain the graph when computing the forces
            # this is why we have to set training to True
            out = model(batch.to_dict(), compute_stress=False, training=True)

            # phonopy convention
            primitive_positions = batch["positions"][0::multiplicity]
            # here, we could use symmtetries to reduce the number of primitive atoms that have to be computed
            # note, that we cannot do everything like usual since autodiff does not work completely independently
            # naiive for loop, optimally use some of the functorch stuff like jacrev, jacfwd or jacobian
            for j in tqdm(range(len(primitive_positions))):
                for k in range(3):
                    # grad_outputs = [torch.ones_like(out["forces"][j, k])]
                    ret_graph = True
                    if j == len(primitive_positions) - 1 and k == 2 and not diffable:
                        ret_graph = False

                    grad = torch.autograd.grad(
                        [out["forces"][j * multiplicity, k]],
                        [batch["positions"]],
                        # grad_outputs=grad_outputs,
                        retain_graph=ret_graph,
                        create_graph=diffable,
                    )[0]
                    if diffable:
                        gradient[j, :, k, :] += grad
                    else:
                        gradient[j, :, k, :] += grad.detach().cpu().numpy()

    gradient /= len(models)
    if diffable:
        gradient = torch.negative(gradient)
    else:
        gradient = -gradient
    return gradient


def compute_phonons(
    structure_file="POSCAR",
    model_file=None,
    backend="torch",
    supercell=[2, 2, 2],
    primitive_matrix=None,
    relax=False,
    relax_fmax=1e-5,
    symprec=1e-5,
    silent=False,
    device="cuda",
    lg_sym=False,
    lg_symprec=1e-2,
    lg_round=None,
    apply_constraints=False,
    autodiff=False,
    dispdist=0.01,
    cueq=False,
) -> Phonopy:
    """
    A function to compute phonons using a given structure file with optional parameters for model file, backend, supercell size, primitive matrix, relaxation, silent mode, and device.
    Returns the Phonopy object.
    """

    supercell = np.array(supercell)
    if len(supercell.ravel()) == 3:
        supercell_matrix = np.diag(np.array(supercell))
    else:
        supercell_matrix = np.array(supercell).reshape((3, 3))
    multiplicity = int(np.round(np.linalg.det(supercell_matrix)))
    if primitive_matrix is not None:
        if len(primitive_matrix) == 9:
            primitive_matrix = np.array(primitive_matrix).reshape((3, 3))
        multiplicity = int(np.round(multiplicity / np.linalg.det(primitive_matrix)))

    print("evaluating supercell:\n", supercell_matrix)

    if isinstance(structure_file, Atoms):
        atoms = structure_file
    else:
        atoms = ase.io.read(structure_file)
    print({s: s for s in atoms.symbols})

    if backend == "jax":
        from mace_jax_extensions.calculators.mace import (
            MACEJAXCalculator,
            make_model_from_pkl,
        )

        model_fn, params, r_max = make_model_from_pkl(model_file)
        calc = MACEJAXCalculator(
            model=model_fn,
            params=params,
            r_max=r_max,
            device="cuda",
            default_dtype="float32",
        )
    elif backend == "torch" and cueq:
        try:
            calc = MACECalculator(model_file, device=device, enable_cueq=True)
        except:
            calc = MACECalculator(model_file, device=device)

    else:
        raise ValueError(f"backend {backend} not known")

    atoms.calc = calc

    if relax:
        optimizer = LBFGS(atoms)
        optimizer.run(fmax=relax_fmax, steps=1000)
        if isinstance(structure_file, Atoms):
            outfname = "POSCAR_relaxed"
        else:
            outfname = f"{structure_file}_relaxed"
        ase.io.write(outfname, atoms, format="vasp")

    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell[:],
        positions=atoms.positions,
    )
    phonon = Phonopy(
        unitcell,
        supercell_matrix=supercell_matrix,
        primitive_matrix=primitive_matrix,
        symprec=symprec,
    )

    if lg_sym:
        site_symmetry, atom_num, perm_ops, rotations = get_site_symmetry(
            atoms,
            phonon,
            symprec=lg_symprec,
            round=lg_round,
        )

        phonon.generate_displacements(
            distance=dispdist, atom_nums=atom_num, site_symmetrys=site_symmetry
        )
        phonon._add_rotations = (rotations,)
        phonon._add_permutation = (perm_ops,)
        phonon._add_sym_tol = (lg_symprec,)
        phonon._add_site_symmetry = (site_symmetry,)
    else:
        phonon.generate_displacements(distance=dispdist)
    supercells = phonon.supercells_with_displacements

    nstructs = len(supercells)

    model_forces = []

    scatoms = ase.Atoms(
        positions=phonon.supercell.positions,
        cell=phonon.supercell.get_cell()[:],
        symbols=phonon.supercell.symbols,
        pbc=[True, True, True],
    )
    # print(supercells[0].symbols)

    if not autodiff:
        scatoms.calc = calc
        for i_struct in tqdm(range(nstructs)):
            # t1 = time.time()
            scatoms.positions = supercells[i_struct].positions
            # scatoms.get_potential_energy()
            dforces = scatoms.get_forces()
            model_forces.append(dforces)
            # print("time taken for displacement", i_struct + 1, time.time() - t1)

        model_forces = np.array(model_forces)

        phonon.set_forces(model_forces)

        print("computing force constants...")
        phonon.produce_force_constants()
    else:
        print("computing force constants via automatic differentiation...")
        phonon.set_force_constants(
            autodiff_fcs(
                scatoms,
                phonon.supercell_matrix,
                primitive_matrix=primitive_matrix,
                calc=calc,
            )
        )

    # ph.file_IO.write_force_constants_to_hdf5(phonon.force_constants)
    if lg_sym or apply_constraints:
        try:
            from swieser_scripts.evaltools.force_constant_correction import (
                get_corrected_force_constants,
            )
        except:
            raise NotImplementedError("constraints not implemented yet")

        if (
            phonon.force_constants.shape[0] == phonon.force_constants.shape[1]
            and multiplicity > 1
        ):
            compact_FCs = full_fc_to_compact_fc(phonon, phonon.force_constants)
            phonon.force_constants = compact_FCs
        # TODO make this more general not just 2
        fcs = get_corrected_force_constants(phonon, direction=2, use_constraints="all")

        phonon.force_constants = fcs

    gammafreqs = phonon.get_frequencies([0, 0, 0])
    with open("gammafreqs.dat", "w") as freqfile:
        for val in gammafreqs:
            freqfile.write("%14.8f  %14.8f\n" % (val, val / 0.0299792458))
            if val < -0.1:
                if not silent:
                    print("WARNING: imaginary frequency " + str(val))
        freqfile.close()

    fcs = phonon.get_force_constants()
    phonon.save("phonopy.yaml")
    # ph.file_IO.write_FORCE_CONSTANTS(fcs, p2s_map=phonon._primitive.p2s_map)
    ph.file_IO.write_force_constants_to_hdf5(fcs, p2s_map=phonon._primitive.p2s_map)
    return phonon

def main():

    parser = argparse.ArgumentParser(
        description="compute phonons using phonopy with the help of MACE models"
    )

    parser.add_argument(
        "--supercell",
        dest="supercell",
        type=int,
        default=[2, 2, 2],
        nargs="*",
        help="supercell matrix for the phonon evaluation",
    )
    parser.add_argument(
        "--primmat",
        dest="primitive_matrix",
        type=float,
        default=None,
        nargs=9,
        help="primitive matrix for the phonon evaluation",
    )
    parser.add_argument(
        "--backend",
        dest="backend",
        type=str,
        default="torch",
        help="torch or jax",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--relax",
        dest="relax",
        action="store_true",
        default=False,
        help="optimize structure before phonon computation",
    )
    parser.add_argument(
        "--lg_sym",
        dest="lg_sym",
        action="store_true",
        default=False,
        help="use line group symmetry",
    )
    parser.add_argument(
        "--lg_symprec",
        dest="lg_symprec",
        type=float,
        default=1e-2,
        help="symmetry precision for line group symmetry",
    )
    parser.add_argument(
        "--apply_constraints",
        dest="apply_constraints",
        action="store_true",
        default=False,
        help="apply sum rules",
    )
    parser.add_argument(
        "--lg_round",
        dest="lg_round",
        type=int,
        default=None,
        help="use this value to round during the line group detection. Default: -log10 of lg_symprec",
    )
    parser.add_argument(
        "--symprec",
        dest="symprec",
        type=float,
        default=1e-5,
        help="symmetry precision for phonopy",
    )
    parser.add_argument(
        "--autodiff",
        dest="autodiff",
        action="store_true",
        default=False,
        help="use automatic differentiation to compute the force constants instead of finite differences",
    )
    parser.add_argument(
        "--ddist",
        dest="dispdist",
        type=float,
        default=0.01,
        help="if automatic differentiation is not used, this defines the finite displacement distance",
    )
    parser.add_argument(
        "model_file",
        default="model.pth",
        nargs="*",
        help="file name for the model parameters. Can also specify multiple files to run in ensemble mode",
    )
    parser.add_argument(
        "--nocueq",
        dest="cueq",
        action="store_true",
        default=True,
        help="do not use cuEquivariance acceleration",
    )

    args = parser.parse_args()

    afp = open("p_cmnd.log", "a")
    afp.write(" ".join(sys.argv) + "\n")
    afp.close()

    compute_phonons(**vars(args))


if __name__ == "__main__":
    main()
