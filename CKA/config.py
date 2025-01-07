import os.path as osp
from pathlib import Path
from multiprocessing import cpu_count

config = {}
config["BASE_DIR"] = Path(osp.dirname(osp.realpath(__file__)))
config["N_CPUS"] = cpu_count() - 1
config["TMP"] = Path("/data/shared/private/mwelsch/tmp")
config["GRAM_MATRICES"] = Path("/data/shared/private/mwelsch/distill_gram_matrices")
assert osp.exists(config["TMP"]), f'{config["TMP"]} does not exist'
assert osp.exists(config["GRAM_MATRICES"]), f'{config["GRAM_MATRICES"]} does not exist'
