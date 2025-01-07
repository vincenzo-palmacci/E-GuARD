import numpy as np
import pandas as pd
import pickle
import os

from datetime import datetime
from pathlib import Path
from scipy.io import savemat
from tqdm import tqdm
from itertools import combinations

from config import config
from trained_pairs.compute_morganfps import compute_morgan_fingerprints
from gram_rf import rf_kernel
from metrics import cka


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def get_rf(path):
    with open(path, "rb") as f:
        rfc = pickle.load(f)
    return rfc


def get_test_data(path):
    test = pd.read_csv(Path(path))
    test["FP"] = test.smiles.apply(compute_morgan_fingerprints)
    return np.array(test["FP"].values.tolist())


def get_gram_from_path(path, features):
    rfc = get_rf(path)
    gram_rf = rf_kernel(rfc, features)
    return {"PATH": str(path), "GRAM_RF": gram_rf}


def main():
    base_path = Path('/data/shared/vin+mat/20241023') / "trained_replicates"
    comparision_teacher = pd.DataFrame(
        {
            "dataset": [],
            "acquisition_func": [],
            "replica": [],
            "iteration": [],
            "cka_rf_to_teacher": [],
        }
    )
    comparision_seeds = pd.DataFrame(
        {
            "dataset": [],
            "acquisition_func": [],
            "iteration": [],
            "replica1": [],
            "replica2": [],
            "inter_replica_cka_rf": [],
        }
    )
    n_replicas = 5
    n_iterations = 10
    try:
        for task in tqdm(os.listdir(base_path), "Task", leave=True, position=0):
            task_timestamp = get_timestamp()
            feature_matrix_test = get_test_data(base_path / f"{task}/{task}.csv")

            teacher_gram = get_gram_from_path(
                base_path / f"{task}/{task}.pkl", features=feature_matrix_test
            )
            filename_teacher_gram = f"{task}_teacher_gram_{task_timestamp}.mat"

            savemat(config["GRAM_MATRICES"] / filename_teacher_gram, teacher_gram)

            for acquisition_func in tqdm(
                os.listdir(base_path / task),
                "Aquisition function",
                leave=False,
                position=1,
            ):
                acquisition_func_path = base_path / task / acquisition_func
                if os.path.isdir(acquisition_func_path):
                    pairwise = {}
                    for replica in range(1, n_replicas + 1):
                        pairwise[replica] = {}
                        for iteration in range(1, n_iterations + 1):
                            gram_current_student_path = (
                                acquisition_func_path
                                / f"rep{replica}"
                                / f"{task}_{iteration}.pkl"
                            )
                            current_student_gram = get_gram_from_path(
                                gram_current_student_path,
                                features=feature_matrix_test,
                            )
                            filename_current_student_gram = f"{task}_{task_timestamp}_student_gram_{acquisition_func}_rep{replica}_iter{iteration}.mat"
                            pairwise[replica][iteration] = current_student_gram[
                                "GRAM_RF"
                            ]

                            savemat(
                                config["GRAM_MATRICES"] / filename_current_student_gram,
                                current_student_gram,
                            )
                            cka_rf = cka(
                                teacher_gram["GRAM_RF"],
                                current_student_gram["GRAM_RF"],
                            )
                            tmp = pd.DataFrame(
                                {
                                    "dataset": [task],
                                    "acquisition_func": [acquisition_func],
                                    "replica": [replica],
                                    "iteration": [iteration],
                                    "cka_rf_to_teacher": [cka_rf],
                                }
                            )
                            comparision_teacher = pd.concat([comparision_teacher, tmp])

                    for iteration in range(1, n_iterations + 1):
                        for replica1, replica2 in combinations(
                            list(range(1, n_replicas + 1)), 2
                        ):
                            gram1 = pairwise[replica1][iteration]
                            gram2 = pairwise[replica2][iteration]
                            cka_rf = cka(gram1, gram2)
                            tmp = pd.DataFrame(
                                {
                                    "dataset": [task],
                                    "acquisition_func": [acquisition_func],
                                    "iteration": [iteration],
                                    "replica1": [replica1],
                                    "replica2": [replica2],
                                    "inter_replica_cka_rf": [cka_rf],
                                }
                            )
                            comparision_seeds = pd.concat([comparision_seeds, tmp])

    except Exception as e:
        print(f"Something went wrong {e}")
    finally:
        timestamp = get_timestamp()
        print(comparision_teacher.to_markdown())
        comparision_teacher.to_csv(
            config["BASE_DIR"] / f"result_csv/comparision_teacher_{timestamp}.csv",
            index=False,
        )

        print(comparision_seeds.to_markdown())
        comparision_seeds.to_csv(
            config["BASE_DIR"] / f"result_csv/comparision_seeds_{timestamp}.csv",
            index=False,
        )


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
