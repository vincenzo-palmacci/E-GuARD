import pandas as pd
import numpy as np
import subprocess
import pickle
import toml
import click
from tqdm import tqdm
from tenacity import retry

import torch

from scipy.stats import norm
from acquisition.epig import get_prob_distribution, epig_from_probs


from imblearn.ensemble import BalancedRandomForestClassifier

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

# Disable rdkit logger
RDLogger.DisableLog("rdApp.*")


def reinvent_generation(task, sampling, version):
    """
    Run REINVENT4 to generate molecules conditioned on the task
    """
    command = f"reinvent v{version}/{sampling}/{task}/config.toml"
    result = subprocess.run(
        command, shell=True, check=True, capture_output=True, text=True
    )

    return result.stdout


def molskill_scoring(task, sampling, version):
    """
    Run the molskill scoring script and return the output
    """

    molskill_scorer = (
        "eGuard/run_molskill/score_smiles.py"  # Path to the molskill scoring script
    )

    command = (
        f"conda run -n molskill python {molskill_scorer} {task} {sampling} {version}"
    )
    result = subprocess.run(
        command, shell=True, check=True, capture_output=True, text=True
    )

    return result.stdout


def preprocess(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
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
    return standard_smiles


def compute_morgan(smile, radius):
    molecule = Chem.MolFromSmiles(smile)
    fp_object = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=2048)
    morgan_fp = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp_object, morgan_fp)
    return morgan_fp


def compute_prob_distributions_for_epig(
    model, scored_molecules, target_subset_size=1000
):
    # Compute Morgan FPs for all molecules in the generated pool (scaffold memory)
    fps_pool = [compute_morgan(s, 3) for s in scored_molecules["SMILES"].tolist()]
    # Define a subset of target molecules on which we want to improve the model e.g., the top 1000 high-scored molecules
    target_molecules = scored_molecules.sort_values(
        "interference", ascending=False
    ).head(target_subset_size)
    # Compute Morgan FPs for the target molecules
    fps_target = [compute_morgan(s, 3) for s in target_molecules["SMILES"].tolist()]
    # Get all probability distributions from the model estimators (trees)
    probs_pool = get_prob_distribution(model, fps_pool)
    probs_target = get_prob_distribution(model, fps_target)
    return probs_pool, probs_target


@retry
def acquire_molecules(model, task, sampling, version):
    """
    Generate and Acquire molecules to be added to the training set.
    """
    print("Generating molecules...")
    reinvent_generation(task, sampling, version)
    torch.cuda.empty_cache()

    def compute_epig(model, scored_molecules, selection_size):
        # Get predicted probability distributions from the model estimators (trees)
        probs_pool, probs_target = compute_prob_distributions_for_epig(
            model, scored_molecules
        )
        # Calculate the epig scores from the probability distributions
        scored_molecules["epig_score"] = epig_from_probs(
            probs_pool, probs_target
        ).tolist()
        # Sort the molecules by the epig score and take the top 500
        scored_molecules = scored_molecules.sort_values(
            "epig_score", ascending=False
        ).head(selection_size)

        return scored_molecules

    if "skill" not in sampling:
        print("Acquiring molecules...")
        # Read reinvent scaffold memory
        scored_molecules = pd.read_csv(
            f"/home/vpalmacci/Projects/FTF4/FtF4/RUN_REINVENT/v{version}/{sampling}/{task}/{task}_1.csv"
        )
        # Preprocess the generated smiles
        scored_molecules["SMILES"] = scored_molecules["SMILES"].apply(preprocess)
        scored_molecules.dropna(inplace=True)

        if sampling == "random":
            # Just randomly sample 250 molecules
            scored_molecules = scored_molecules.sample(250)

        elif sampling == "greedy":
            # Sort the molecules by the interference score and take the top 250
            scored_molecules = scored_molecules.sort_values(
                "interference", ascending=False
            ).head(250)

        elif sampling == "epig":
            # Compute the EPIG score
            scored_molecules = compute_epig(model, scored_molecules, 250)

    else:
        # Run the molskill scoring script
        # Preprocessing happens in the molskill scoring script
        print("Scoring molecules...")

        try:
            molskill_scoring(task, sampling, version)
        except:
            # just rerun the iteration
            print("Molskill scoring failed! Rerunning the last iteration!")
            acquire_molecules(task, sampling, version)

        # Read the script output
        scored_molecules = pd.read_csv(
            f"/home/vpalmacci/Projects/FTF4/FtF4/RUN_REINVENT/v{version}/{sampling}/{task}.csv"
        )

        if sampling == "greedyskill":
            # Consider only molecuels with negative molskill score
            # Sort the molecules by the molskill score and take the top 250
            # scored_molecules["combined"] = scored_molecules["interference"].sort_va * scored_molecules["molskill"]
            scored_molecules = scored_molecules.sort_values(
                "interference", ascending=False
            ).head(500)
            scored_molecules = scored_molecules.sort_values(
                "molskill", ascending=True
            ).head(250)

        elif sampling == "epigskill":
            # Compute the EPIG score
            scored_molecules = compute_epig(model, scored_molecules, selection_size=500)
            # scored_molecules["combined"] = scored_molecules["epig_score"] * scored_molecules["molskill"]
            scored_molecules = scored_molecules.sort_values(
                "epig_score", ascending=False
            ).head(500)
            scored_molecules = scored_molecules.sort_values(
                "molskill", ascending=True
            ).head(250)

        print("Molecules scored! Just read the csv!")

        # Read the scored molecules and take the first n molecules
        # scored_molecules = scored_molecules.head(250)

    return scored_molecules


@click.command()
@click.option(
    "-d", "--dataset", required=True, help="Specify the task {fluc, nluc, redox, thiol}"
)
@click.option(
    "-i",
    "--iteration",
    type=int,
    required=True,
    help="Specify the numbe of iterations to run",
)
@click.option(
    "-s",
    "--sampling",
    required=True,
    help="Specify the sampling strategy {random, greedy, molskill, greedyskill}",
)
@click.option("-v", "--version", type=int, required=True, help="Specify the version")
def Main(sampling, dataset, iteration, version):
    """
    Define the self training + active learning loop.
    """
    # generated_fps = [] # List to store the generated fingerprints at each iteration
    print("Loading training data...")
    task = dataset.split(".")[0]
    training_data = pd.read_csv(f"data/train/{task}.csv")

    training_smiles = training_data["smiles"].values
    training_labels = training_data["label"].values.tolist()

    # Compute fingerprints for the training set
    training_fps = [compute_morgan(smi, 3) for smi in tqdm(training_smiles)]

    for iter in range(iteration):
        print(f"Iteration {iter+1}:")
        # Load the trained model that is used for scoring the reinvent run
        print("Reading the toml file...")
        toml_file = f"v{version}/{sampling}/{task}/config.toml"
        with open(toml_file, "r") as f:
            toml_string = f.read()

        parsed_toml = toml.loads(toml_string)
        model_file = parsed_toml["stage"][0]["scoring"]["component"][0][
            "ExternalModel"
        ]["endpoint"][0]["params"]["model_file"]
        with open(model_file, "rb") as f:
            trained_model = pickle.load(f)
        ### Step 1 and Step 2 are combined in the acquire_molecules function and avoid faiilures.
        ### Step 1: Run REINVENT4 to generate molecules conditioned on the task
        ### Step 2: Select molecules to be added to the training set.
        scored_molecules = acquire_molecules(trained_model, task, sampling, version)

        # Save the molecules selected for retraining
        scored_molecules.to_csv(
            f"/home/vpalmacci/Projects/FTF4/FtF4/RUN_REINVENT/v{version}/{sampling}/{task}/selected_{iter+1}.csv",
            index=False,
        )

        processed_smiles = scored_molecules["SMILES"].values
        generated_fps = [compute_morgan(smi, 3) for smi in tqdm(processed_smiles)]

        generated_labels = (
            (scored_molecules["interference"] >= 0.5).astype(int).values.tolist()
        )

        ### Step 3: Append the generated data to the training set
        training_fps += generated_fps
        training_labels += generated_labels
        print(
            "Percentage of positive samples in the training set: ",
            sum(training_labels) / len(training_labels),
        )  # Print the ratio of positive samples in the training set

        ### Step 4: Retrain the model with the augmented data
        # Load the trained model
        print("Retraining the model...")
        if iter == 0:
            model_path = f"eGuard/teacher/trained_models/{task}.pkl"
        else:
            model_path = f"/home/vpalmacci/Projects/FTF4/FtF4/RUN_REINVENT/v{version}/{sampling}/{task}/{task}_{iter}.pkl"

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Retrain the model
        # Get suggested hyperparameters.
        hyperparameters = np.load(
            f"eGuard/teacher/hyperparameters/{task}.npy", allow_pickle=True
        )[()]

        # Instanciate the random forest classifier with the suggested hyperparameters.
        print(hyperparameters)

        model = BalancedRandomForestClassifier(
            n_estimators=hyperparameters["n_estimators"],
            max_depth=hyperparameters["max_depth"],
            min_samples_split=hyperparameters["min_samples_split"],
            max_features=hyperparameters["max_features"],
            bootstrap=True,
            n_jobs=16,
        )

        model.fit(training_fps, training_labels)

        # Save the retrained model
        with open(
            f"/home/vpalmacci/Projects/FTF4/FtF4/RUN_REINVENT/v{version}/{sampling}/{task}/{task}_{iter+1}.pkl",
            "wb",
        ) as f:
            pickle.dump(model, f)

        ### Step 5: Update toml file with new scoring model and checkpoint
        # Load the toml file.
        print("Updating the toml file...")
        toml_file = f"v{version}/{sampling}/{task}/config.toml"

        with open(toml_file, "r") as f:
            toml_string = f.read()

        parsed_toml = toml.loads(toml_string)

        # Update the agent file.
        parsed_toml["parameters"][
            "agent_file"
        ] = f"/home/vpalmacci/Projects/FTF4/FtF4/RUN_REINVENT/v{version}/{sampling}/{task}/{task}.chkpt"
        # Update the scoring model to be loaded.
        parsed_toml["stage"][0]["scoring"]["component"][0]["ExternalModel"]["endpoint"][
            0
        ]["params"][
            "model_file"
        ] = f"/home/vpalmacci/Projects/FTF4/FtF4/RUN_REINVENT/v{version}/{sampling}/{task}/{task}_{iter+1}.pkl"  ###ADD THIS FOLDER

        # Write the updated config file.
        with open(
            f"/home/vpalmacci/Projects/FTF4/FtF4/RUN_REINVENT/v{version}/{sampling}/{task}/config.toml",
            "w",
        ) as f:
            toml.dump(parsed_toml, f)

        # Copy the checkpoint file to the correct location
        checkpoint_file = f"/home/vpalmacci/Projects/FTF4/FtF4/RUN_REINVENT/v{version}/{sampling}/{task}/chkpt/{task}_{iter+1}.chkpt"
        command = f"cp /home/vpalmacci/Projects/FTF4/FtF4/RUN_REINVENT/v{version}/{sampling}/{task}/{task}.chkpt {checkpoint_file}"
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )

        print(f"Iteration {iter+1} completed!\n")

    print("Self-training loop completed!")
    return None


if __name__ == "__main__":
    Main()
