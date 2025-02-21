import os

import fegrow
import prody
from dask.distributed import LocalCluster
from fegrow import ChemSpace, RGroups
from rdkit import Chem


def scoring_function(rmol, pdb_filename, data):
    affinities = rmol.gnina(receptor_file=pdb_filename)
    return min(affinities.CNNaffinity)


def optimize_rgroups(
    ligand_filename, protein_filename, attachment_index, output_dir, use_ani=False
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rgroups = RGroups()

    init_mol = Chem.SDMolSupplier(ligand_filename, removeHs=False)[0]
    init_mol = Chem.AddHs(init_mol)
    for atom in init_mol.GetAtoms():
        if atom.GetIdx() == attachment_index:
            atom.SetAtomicNum(0)
            break

    template = fegrow.RMol(init_mol)

    rec = prody.parsePDB(protein_filename).select("not (nucleic or hetatm or water)")
    prody.writePDB("rec.pdb", rec)
    fegrow.fix_receptor("rec.pdb", "rec_final.pdb")
    prody.parsePDB("rec_final.pdb")

    cluster = LocalCluster(n_workers=8)
    cs = ChemSpace(dask_cluster=cluster)
    cs.add_scaffold(template)
    cs.add_protein("rec_final.pdb")

    for mol in rgroups.Mol:
        mol = Chem.AddHs(mol)
        cs.add_rgroups(mol)

    results = cs.evaluate(
        scoring_function=scoring_function,
        num_conf=50,
        minimum_conf_rms=0.5,
        use_ani=use_ani,
        platform="cuda",
        gnina_gpu=True,
        min_dst_allowed=0.5,
    )

    with Chem.SDWriter(os.path.join(output_dir, "results.sdf")) as writer:
        for row in results.sort_values("score", ascending=False).iterrows():
            mol = row[1]["Mol"]
            mol.SetProp("score", str(row[1]["score"]))
            writer.write(mol)

    best_molecule_index = results["score"].idxmin()
    best_molecule = results.loc[best_molecule_index, "Mol"]

    print(
        f"Best molecule (index: {best_molecule_index}): {Chem.MolToSmiles(best_molecule)}"
    )
    print(f"Score: {results.loc[best_molecule_index, 'score']}")

    best_molecule.to_file(os.path.join(output_dir, "best_molecule.pdb"))

    results.sort_values("score", ascending=False).to_csv(
        os.path.join(output_dir, "results.csv")
    )

    if not os.path.exists(os.path.join(output_dir, "results")):
        os.makedirs(os.path.join(output_dir, "results"))
    for i, mol in enumerate(results.sort_values("score", ascending=False)["Mol"]):
        mol.to_file(os.path.join(output_dir, "results", f"{i}.pdb"))


def dock_original_ligand(ligand_filename, protein_filename):
    ligand = Chem.SDMolSupplier(ligand_filename, removeHs=False)[0]
    ligand = Chem.AddHs(ligand)

    rec = prody.parsePDB(protein_filename).select("not (nucleic or hetatm or water)")
    prody.writePDB("rec.pdb", rec)
    fegrow.fix_receptor("rec.pdb", "rec_final.pdb")
    prody.parsePDB("rec_final.pdb")

    rmol = fegrow.RMol(ligand)

    affinities = rmol.gnina(receptor_file="rec_final.pdb")
    print(affinities)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize R-groups for a ligand attached to a protein"
    )
    parser.add_argument("--ligand", type=str, help="Ligand filename")
    parser.add_argument("--protein", type=str, help="Protein filename")
    parser.add_argument(
        "--attachment_index", type=int, help="Attachment index of atom in ligand"
    )
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--use_ani", type=bool, default=False, help="Whether to use TorchANI"
    )
    args = parser.parse_args()

    optimize_rgroups(
        ligand_filename=args.ligand,
        protein_filename=args.protein,
        attachment_index=args.attachment_index,
        output_dir=args.output_dir,
        use_ani=args.use_ani,
    )
