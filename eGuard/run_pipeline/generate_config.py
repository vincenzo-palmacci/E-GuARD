import toml
import click
import os

# TODO: solve this import: from FtF4.path import teacher


@click.command()
@click.option(
    "-d", "--dataset", help="Specify the dataset to generate config file for."
)
@click.option(
    "-s", "--sampling", help="Specify the sampling strategy {random, greedy, hintl}"
)
@click.option("-v", "--version", help="Specify the version of the pipeline.")
def generate_config(dataset, sampling, version):
    # Load the template toml file.
    with open("configs/template.toml", "r") as f:
        toml_string = f.read()

    parsed_toml = toml.loads(toml_string)

    task = dataset.split(".")[0]

    # Create the directory for the task if it does not exist.
    task_dir = f"v{version}/{sampling}/{task}"
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    # Create the tb logs directory for the task if it does not exist.
    tb_logdir = f"v{version}/{sampling}/{task}/tb_logs/"
    if not os.path.exists(tb_logdir):
        os.makedirs(tb_logdir)
    # Specify the tb_logdir in the config file.
    parsed_toml["tb_logdir"] = tb_logdir

    # Create directory for checkpoint files.
    chkpt_dir = f"v{version}/{sampling}/{task}/chkpt/"
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    # Rename the scaffold memory file.
    parsed_toml["parameters"][
        "summary_csv_prefix"
    ] = f"v{version}/{sampling}/{task}/{task}"
    # Rename the checkpoint file.
    parsed_toml["stage"][0]["chkpt_file"] = f"v{version}/{sampling}/{task}/{task}.chkpt"

    # Write the correct scoring model to be loaded.
    model_dir = "/home2/vpalmacci/Projects/FTF4/FtF4/teacher/trained_models"  # TODO: Change this using the path.py file.

    parsed_toml["stage"][0]["scoring"]["component"][0]["ExternalModel"]["endpoint"][0][
        "params"
    ]["model_file"] = f"{model_dir}/{task}.pkl"

    # Write the config file.
    with open(f"{task_dir}/config.toml", "w") as f:
        toml.dump(parsed_toml, f)


if __name__ == "__main__":
    generate_config()
