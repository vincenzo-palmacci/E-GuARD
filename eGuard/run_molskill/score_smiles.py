import pandas as pd
import sys
from tqdm import tqdm
import torch

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger  

from molskill.scorer import MolSkillScorer

# Disable rdkit logger
RDLogger.DisableLog('rdApp.*')

def preprocess(smile):
    try:
        mol = Chem.MolFromSmiles(smile, sanitize=True)
        mol = rdMolStandardize.Cleanup(mol)
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        te = rdMolStandardize.TautomerEnumerator()
        mol = te.Canonicalize(mol)
    except:
        return None
    try:
        standard_smiles = rdMolStandardize.StandardizeSmiles(Chem.MolToSmiles(mol))
    except:
        return None
    smi = standard_smiles
    return smi

def Main(task, sampling, version):
    # Load the data.
    print("Loading data...")
    reinvent_memory = pd.read_csv(f"/home/vpalmacci/Projects/FTF4/FtF4/RUN_REINVENT/v{version}/{sampling}/{task}/{task}_1.csv")

    print("Standardizing data...")
    # Generate standardized SMILES and collect interference scores.
    reinvent_memory["SMILES"] = reinvent_memory["SMILES"].apply(preprocess)
    reinvent_memory.dropna(inplace=True)

    # Score the SMILES.
    print("Scoring data...")
    scorer = MolSkillScorer()
    reinvent_memory["molskill"] = scorer.score(reinvent_memory["SMILES"].values.tolist())

    # Rank the SMILES based on the MolSkill score.
    print("Ranking data...")
    reinvent_memory = reinvent_memory.sort_values("molskill", ascending=True)

    # Save the data.
    reinvent_memory.to_csv(f"/home/vpalmacci/Projects/FTF4/FtF4/RUN_REINVENT/v{version}/{sampling}/{task}.csv", index=False)
    # clen torch cache
    torch.cuda.empty_cache()
    
    return reinvent_memory


if __name__ == "__main__":
    # Specify arguments.
    task = sys.argv[1]
    sampling = sys.argv[2]
    version = int(sys.argv[3])
    # Run the main function.
    Main(task, sampling, version)
