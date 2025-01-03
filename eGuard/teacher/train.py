import pandas as pd
import numpy as np
from tqdm import tqdm
import click
import os.path as osp
import pickle

# Importing required libraries for machine learning
from imblearn.ensemble import BalancedRandomForestClassifier

# Importing cheminformatics libraries
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, DataStructs

from FtF4.path import training

# Ignoring warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Disable rdkit logger
RDLogger.DisableLog('rdApp.*')

"""
Random Forest Classifier: Train models.
"""

def _compute_morgan(smile, radius):
    """
    Function to compute morgan fingerprints for a list of smiles.
    """
    molecule = Chem.MolFromSmiles(smile)
    fp_object = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=2048)
    morgan_fp = np.zeros((0, ))
    DataStructs.ConvertToNumpyArray(fp_object, morgan_fp)
    
    return morgan_fp

@click.command()
@click.option("-d", "--dataset", required=True, help="Specify the dataset {fluc, nluc, redox, thiol}")
def Main(dataset):
    """
    Random Forest Classifier: Train models.
    """
    # Load data.
    print("\n Loading data ...")
    dataname = dataset.split(".")[0]
    data = pd.read_csv(f"/home/vpalmacci/Projects/FTF4/data/train/{dataset}")
    smiles = data["smiles"].values # get samples
    # Get labels.
    labels = data["label"].values

    # Compute molecules descriptors.
    fingerprints = np.array([_compute_morgan(i, radius=3) for i in tqdm(smiles)], dtype=float)

    print("\nDataset specs: ")
    print("\t# Compound:", labels.shape[0])
    print("\t# features:", fingerprints.shape[1])
    
    X, y = fingerprints, labels

    # Get suggested hyperparameters.
    hyperparameters = np.load(osp.join(f"results/validation/{dataname}.npy"), allow_pickle=True)[()]

    # Instanciate the random forest classifier with the suggested hyperparameters.
    print(hyperparameters)

    model = BalancedRandomForestClassifier(
                n_estimators = hyperparameters["n_estimators"],
                max_depth = hyperparameters["max_depth"],
                min_samples_split = hyperparameters["min_samples_split"],
                max_features = hyperparameters["max_features"],
                bootstrap = True,
                n_jobs = 32,
                )

    # Train and validate.
    model.fit(X, y)

    # Save trained model.
    with open(f"trained_models/{dataname}.pkl", "wb") as f:
        pickle.dump(model, f)

    return print("Model saved")

if __name__=="__main__":
    Main()
