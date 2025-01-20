import pandas as pd
import numpy as np
from tqdm import tqdm
import click
import os
import pickle

# Importing required libraries for machine learning
from imblearn.ensemble import BalancedRandomForestClassifier

# Importing cheminformatics libraries
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, DataStructs

# Ignoring warnings for cleaner output
import warnings

warnings.filterwarnings("ignore")

# Disable rdkit logger
RDLogger.DisableLog("rdApp.*")

"""
Random Forest Classifier: Train models.
"""

# Seeding
np.random.seed(13)

# Directories
datadir = "../../data"

def _compute_morgan(smile, radius):
    """
    Function to compute morgan fingerprints for a list of smiles.
    """
    molecule = Chem.MolFromSmiles(smile)
    fp_object = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=2048)
    morgan_fp = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp_object, morgan_fp)

    return morgan_fp


@click.command()
@click.option("-s", "--source", required=False, help="Specify the source {alves, polaris}", type=str)
@click.option("-d", "--dataset", required=True, help="Specify the dataset {fluc, nluc, redox, thiol}", type=str)
def Main(source, dataset):
    """
    Random Forest Classifier: Train models.
    """
    # Load data.
    print("\n Loading data ...")
    dataname = dataset.split(".")[0]

    if not source:
        source = "alves"
    
    data = pd.read_csv(f"{datadir}/{source}/train/{dataname}.csv")
    smiles = data["smiles"].values  # get samples
    # Get labels.
    labels = data["label"].values

    # Compute molecules descriptors.
    fingerprints = np.array(
        [_compute_morgan(i, radius=3) for i in tqdm(smiles)], dtype=float
    )

    print("\nDataset specs: ")
    print("\t# Compound:", labels.shape[0])
    print("\t# features:", fingerprints.shape[1])

    X, y = fingerprints, labels

    # Get suggested hyperparameters.
    hyperparameters = np.load(
        os.path.join(f"hyperparameters/{dataname}.npy"),
        allow_pickle=True,
    )[()]

    # Instanciate the random forest classifier with the suggested hyperparameters.
    print(hyperparameters)

    model = BalancedRandomForestClassifier(
        n_estimators=hyperparameters["n_estimators"],
        max_depth=hyperparameters["max_depth"],
        min_samples_split=hyperparameters["min_samples_split"],
        max_features=hyperparameters["max_features"],
        bootstrap=True,
        n_jobs=32,
    )

    # Train and validate.
    model.fit(X, y)

    # Save trained model.
    if not os.path.exists("trained_models"):
        os.makedirs("trained_models")
    with open(f"trained_models/{dataname}.pkl", "wb") as f:
        pickle.dump(model, f)

    return print("Model saved")


if __name__ == "__main__":
    Main()
