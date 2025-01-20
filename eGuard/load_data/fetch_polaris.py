import pandas as pd
import numpy as np

import polaris as po
from polaris.hub.client import PolarisHubClient, PolarisFileSystem


def fetch_polaris_benchmarks():
    # Single task classification benchmarks

    benchmarks = [
        "polaris/pkis2-egfr-wt-c-1",
        "polaris/pkis2-ret-wt-c-1",
        "polaris/pkis2-kit-wt-c-1",
        "polaris/pkis2-kit-wt-cls-v2",
        "polaris/pkis2-ret-wt-cls-v2",
        "tdcommons/ames",
    ]

    # Fetch the benchmarks
    for benchmark in benchmarks:
        data = po.load_benchmark(benchmark)
        bname = benchmark.split("/")[1]

        # Load and split the data
        train, test = data.get_train_test_split()
        
        train_df = pd.DataFrame([train.inputs, train.targets.astype(int)]).T
        train_df.columns = ["smiles", "label"]

        test_df = pd.DataFrame([test.inputs], index=["smiles"]).T

        # Save the data
        train_df.to_csv(f"../../data/polaris/train/{bname}.csv", index=False)
        test_df.to_csv(f"../../data/polaris/test/{bname}.csv", index=False)


if __name__ == "__main__":
    fetch_polaris_benchmarks()
