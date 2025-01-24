import os

import fegrow
import prody
from fegrow import ChemSpace, RGroups
from rdkit import Chem


def scoring_function(rmol, pdb_filename, data):
    affinities = rmol.gnina(receptor_file=pdb_filename)
    return min(affinities.CNNaffinity)


def optimize_rgroups(ligand_filename, protein_filename, attachment_index, output_dir):
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

    cs = ChemSpace()
    cs.add_scaffold(template)
    cs.add_protein("rec_final.pdb")

    for mol in rgroups.Mol:
        mol = Chem.AddHs(mol)
        cs.add_rgroups(mol)

    results = cs.evaluate(
        scoring_function=scoring_function,
        num_conf=50,
        minimum_conf_rms=0.5,
        use_ani=True,
        platform="cuda",
        gnina_gpu=True,
    )

    with Chem.SDWriter(os.path.join(output_dir, "results.sdf")) as writer:
        for row in results.sort_values("score").iterrows():
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

    results.to_csv(os.path.join(output_dir, "results.csv"))
    for i, mol in enumerate(results.sort_values("score")["Mol"]):
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index", type=int)
    args = parser.parse_args()

    data_info = [
        {
            "ligand_filename": "data/sarscov2/coreh.sdf",
            "protein_filename": "data/sarscov2/7L10.pdb",
            "attachment_index": 40,
            "output_dir": "data/sarscov2/output",
        },
        {
            "ligand_filename": "data/JNK1/ligand_reduced.sdf",
            "protein_filename": "data/JNK1/2GMX.pdb",
            "attachment_index": 29,
            "output_dir": "data/JNK1/output",
        },
        {
            "ligand_filename": "data/BACE1/ligand_reduced.sdf",
            "protein_filename": "data/BACE1/4djw.pdb",
            "attachment_index": 32,
            "output_dir": "data/BACE1/output",
        },
        {
            "ligand_filename": "data/BACE1/ligand_reduced_2.sdf",
            "protein_filename": "data/BACE1/4djw.pdb",
            "attachment_index": 21,
            "output_dir": "data/BACE1/output_2",
        },
    ]
    optimize_rgroups(**data_info[args.data_index])
    # dock_original_ligand("data/JNK1/ligand_reduced.sdf", "data/JNK1/2GMX.pdb")
