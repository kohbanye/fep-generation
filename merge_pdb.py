from rdkit import Chem
import os
from natsort import natsorted

with Chem.SDWriter("results.sdf") as writer:
    for filename in natsorted(os.listdir("results")):
        path = os.path.join("results", filename)
        mol = Chem.MolFromPDBFile(path)
        if mol is None:
            continue
        writer.write(mol)

