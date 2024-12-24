import pandas as pd
import numpy as np
from tqdm import tqdm
import click
import os.path as osp
import pickle

# Importing required libraries for machine learning
#from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, matthews_corrcoef, roc_auc_score, average_precision_score, balanced_accuracy_score

# Importing cheminformatics libraries
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Ignoring warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

"""
Random Forest Classifier: Validation with hyperparameters search.
"""
# Global variables:
datadir = "../data/test/"
outdir = "results/predictions/"

# Seeding
np.random.seed(13)

def _compute_morgan(smile, radius):
    """
    Function to compute morgan fingerprints for a list of smiles.
    """
    molecule = Chem.MolFromSmiles(smile)
    fp_object = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=2048)
    morgan_fp = np.zeros((0, ))
    DataStructs.ConvertToNumpyArray(fp_object, morgan_fp)
    
    return morgan_fp


def _classification_report(true, preds):
    """
    Make classification report.
    """
    classification_metrics = {"recall":0, "precision":0, "mcc":0, "auroc":0, "aupr":0, "ba":0}
    
    binary = [1 if i >= 0.5 else 0 for i in preds]

    classification_metrics["recall"] = recall_score(true, binary)
    classification_metrics["precision"] = precision_score(true, binary)
    classification_metrics["mcc"] = matthews_corrcoef(true, binary)
    classification_metrics["ba"] = balanced_accuracy_score(true, binary)
    classification_metrics["auroc"] = roc_auc_score(true, preds)
    classification_metrics["aupr"] = average_precision_score(true, preds)
    
    return classification_metrics


@click.command()
@click.option("-d", "--dataset", required=True, help="Check available datasets in data/raw/")
def Main(dataset):
    """
    Random Forest Classifier: validation with hyperparameters search.
    """
    # Load data.
    print("\n Loading data ...")

    data = pd.read_csv(osp.join(datadir, f"{dataset}.csv"))
    smiles = data["Clean_Smiles"].values
    labels = data["labels"].values

    # Compute molecules descriptors.
    fingerprints = np.array([_compute_morgan(i, radius=2) for i in tqdm(smiles)], dtype=float)

    print("\nDataset specs: ")
    print("\t# Compound:", labels.shape[0])
    print("\t# features:", fingerprints.shape[1])

    # Load trained model.
    with open(f"models/{dataset}.pkl", "rb") as inp:
        model = pickle.load(inp)
    # Predict test data.
    predictions = model.predict_proba(fingerprints)[:,1]

    # Save predictions.
    with open(osp.join(outdir, f"{dataset}.csv"), "w") as outfi:
        frame = pd.DataFrame([smiles, predictions]).T
        frame.columns = ["Smiles", "Probability"]
        frame.to_csv(outfi)

    # Compute performances.
    performances = _classification_report(labels, predictions)

    return print(performances)

if __name__=="__main__":
    Main()
