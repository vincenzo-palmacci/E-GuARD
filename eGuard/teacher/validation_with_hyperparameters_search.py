import pandas as pd
import numpy as np
from tqdm import tqdm
import click
import os

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
import optuna

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
Random Forest Classifier: Validation with hyperparameters search.
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


def _suggest_hyperparameters(trial: optuna.trial.Trial) -> list:
    """
    Suggest hyperparameters for optuna search.
    """
    # Suggest number of trees.
    n_estimators = trial.suggest_int("n_estimators", low=32, high=500, step=128)
    # Suggest trees depth.
    max_depth = trial.suggest_int("max_depth", low=16, high=1024, step=16)
    # Suggest minimum samples split.
    min_samples_split = trial.suggest_int("min_samples_split", low=2, high=5, step=1)
    # Suggest maximum number of features per split.
    max_features = trial.suggest_int("max_features", low=2, high=2048, step=1)

    print(f"Suggested hyperparameters: \n{(trial.params)}")
    return trial.params


def objective(trial: optuna.trial.Trial, X, y) -> float:
    """
    Search for optimal set of hyperparameters via cross validation.
        n_folds = 5
    """
    # Get suggested hyperparameters.
    hyperparameters = _suggest_hyperparameters(trial)
    print(hyperparameters)

    # Run 5 fold cross validation.
    cv = StratifiedKFold(n_splits=5)

    scores = []
    for (
        train_idx,
        test_idx,
    ) in cv.split(X, y):
        # Get training and validation sets.
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Instanciate the random forest classifier with the suggested hyperparameters.
        model = BalancedRandomForestClassifier(
            n_estimators=hyperparameters["n_estimators"],
            max_depth=hyperparameters["max_depth"],
            min_samples_split=hyperparameters["min_samples_split"],
            max_features=hyperparameters["max_features"],
            bootstrap=True,
            n_jobs=32,
        )

        # Fit training data.
        model.fit(X_train, y_train)
        # Compute prediction.
        y_pred = model.predict(X_test)
        # Compute performances.
        metric = matthews_corrcoef(y_test, y_pred)
        scores.append(metric)

    return np.mean(scores)


@click.command()
@click.option("-s", "--source", required=False, help="Specify the source {alves, polaris}", type=str)
@click.option("-d", "--dataset", required=True, help="Specify the dataset {fluc, nluc, redox, thiol}", type=str)
def Main(source, dataset):
    """
    Random Forest Classifier: validation with hyperparameters search.
    """
    # Load data.
    print("\n Loading data ...")
    dataname = dataset.split(".")[0]
    
    if not source:
        source = "alves"
    
    data = pd.read_csv(f"{datadir}/{source}/train/{dataname}.csv")
    smiles = data["smiles"].values  # get samples
    # Get labels.
    labels = data["label"].values.astype(int)

    # Compute molecules descriptors.
    fingerprints = np.array(
        [_compute_morgan(i, radius=3) for i in tqdm(smiles)], dtype=float
    )

    print("\nDataset specs: ")
    print("\t# Compound:", fingerprints.shape[0])
    print("\t# features:", fingerprints.shape[1])

    # Run hyperparameters search with optuna.
    study_name = "_".join([dataset, "optuna_run"])
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(),
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(lambda trial: objective(trial, fingerprints, labels), n_trials=50)

    best_trial = study.best_trial.params

    # Save best hyperparameters.
    if not os.path.exists("hyperparameters"):
        os.makedirs("hyperparameters")

    np.save(os.path.join("hyperparameters", f"{dataname}.npy"), best_trial)


if __name__ == "__main__":
    Main()
