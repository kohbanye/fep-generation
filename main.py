import fegrow
import prody
from fegrow import ChemSpace, Linkers, RGroups
from rdkit import Chem


def scoring_function(rmol, pdb_filename, data):
    affinities = rmol.gnina(receptor_file=pdb_filename)
    return min(affinities.CNNaffinity)


rgroups = RGroups()
linkers = Linkers()

init_mol = Chem.SDMolSupplier("data/sarscov2/coreh.sdf", removeHs=False)[0]
attachment_index = 40
for atom in init_mol.GetAtoms():
    if atom.GetIdx() == attachment_index:
        atom.SetAtomicNum(0)
        break

template = fegrow.RMol(init_mol)


rec = prody.parsePDB("data/7L10.pdb").select("not (nucleic or hetatm or water)")
prody.writePDB("rec.pdb", rec)
fegrow.fix_receptor("rec.pdb", "rec_final.pdb")
rec_final = prody.parsePDB("rec_final.pdb")


cs = ChemSpace()
cs.add_scaffold(template)
cs.add_protein("rec_final.pdb")


# all_rgroup_smiles = rgroups.Name.tolist()
# cs.add_smiles(all_rgroup_smiles, h=attachment_index)
for mol in rgroups.Mol:
    cs.add_rgroups(mol)

results = cs.evaluate(
    scoring_function=scoring_function,
    num_conf=50,
    minimum_conf_rms=0.5,
    use_ani=True,
    platform="CPU",
)

print(results)

best_molecule_index = results["score"].idxmin()
best_molecule = results.loc[best_molecule_index, "Mol"]

print(
    f"Best molecule (index: {best_molecule_index}): {Chem.MolToSmiles(best_molecule)}"
)
print(f"Score: {results.loc[best_molecule_index, 'score']}")

best_molecule.to_file("best_molecule.pdb")
