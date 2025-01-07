from sklearn.tree import _tree
import numpy as np
from joblib import Parallel, delayed, dump, load
from multiprocessing import Manager, Process
import gc
from scipy import sparse
import os
from itertools import combinations_with_replacement

from config import config


def Difference(l1: list, l2: list):
    return list(set(l1).difference(set(l2)))


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def next_partition(tree, leaves, depth, partition_ids):
    if depth == 0:
        return partition_ids
    new_partition_ids = []
    for node_id in partition_ids:
        if node_id in leaves:
            new_partition_ids += [node_id]
        else:
            new_partition_ids += [
                tree.tree_.children_left[node_id],
                tree.tree_.children_right[node_id],
            ]
    return next_partition(tree, leaves, depth - 1, new_partition_ids)


def get_partition(tree, depth):
    partition_id = [0]
    leaves = [
        node_id
        for node_id in range(tree.tree_.node_count)
        if tree.tree_.children_left[node_id] == _tree.TREE_LEAF
        and tree.tree_.children_right[node_id] == _tree.TREE_LEAF
    ]
    depth = depth % np.max(tree.tree_.compute_node_depths())
    return next_partition(tree, leaves, depth, partition_id)


def parallel_node_ids(forest, tree_indices, n_samples):
    node_ids = Manager().dict()

    def add_node_id(tree_nr, depth, node_ids):
        node_ids[tree_nr] = get_partition(forest[tree_nr], depth)

    jobs = [
        Process(target=add_node_id, args=(tree_nr, depth, node_ids))
        for tree_nr, depth in zip(
            tree_indices, np.random.randint(low=0, high=10000, size=n_samples)
        )
    ]
    _ = [p.start() for p in jobs]
    _ = [p.join() for p in jobs]
    id_nodes = {key: val for key, val in node_ids.items()}
    dump(id_nodes, os.path.join(config["TMP"], "id_nodes"), compress=3)
    del id_nodes
    _ = gc.collect()


def parallel_partition_desc(forest, tree_indices, features):
    def add_partition_desc(h):
        path_samples = os.path.join(config["TMP"], str(h))
        samples = load(path_samples)
        forest = load(os.path.join(config["TMP"], "forest"))
        tree_indices = load(os.path.join(config["TMP"], "tree_indices"))
        id_nodes = load(os.path.join(config["TMP"], "id_nodes"))
        out = {
            tuple(sample): sparse.csr_array(
                np.concatenate(
                    tuple(
                        forest[tree_nr]
                        .decision_path(sample.reshape(1, -1))
                        .toarray()[0, id_nodes.get(tree_nr)]
                        for tree_nr in tree_indices
                    ),
                    axis=0,
                ).ravel()
            )
            for sample in samples
        }
        dump(out, path_samples, compress=3)
        del out, samples, forest, id_nodes, tree_indices
        _ = gc.collect()
        return h

    hashes_samples = [
        hash(samples.data.tobytes())
        for samples in chunks(features, int(features.shape[0] / config["N_CPUS"]))
    ]
    for h, samples in zip(
        hashes_samples,
        chunks(features, int(features.shape[0] / config["N_CPUS"])),
    ):
        dump(samples, os.path.join(config["TMP"], str(h)), compress=3)
    dump(forest, os.path.join(config["TMP"], "forest"), compress=3)
    dump(tree_indices, os.path.join(config["TMP"], "tree_indices"), compress=3)

    partition_hashes = Parallel(n_jobs=config["N_CPUS"], verbose=0)(
        delayed(add_partition_desc)(h) for h in hashes_samples
    )

    partition_descriptor = {}
    for h in hashes_samples:
        temp = load(os.path.join(config["TMP"], str(h)))
        partition_descriptor.update(temp)
    dump(
        partition_descriptor,
        os.path.join(config["TMP"], "partition_descriptor"),
        compress=3,
    )
    del partition_descriptor, temp
    _ = gc.collect()


def parallel_gram(features, n_samples):
    def calc_gram_row(combs):
        partition_descriptor = load(os.path.join(config["TMP"], "partition_descriptor"))
        gram = np.zeros((len(features), len(features)))
        for comb in combs:
            (index1, sample1), (index2, sample2) = comb
            m1 = partition_descriptor[tuple(sample1)]
            m2 = partition_descriptor[tuple(sample2)]
            kernel = np.sum(m1.indices == m2.indices) / n_samples
            gram[index1, index2] = kernel
            gram[index2, index1] = kernel
        h = hash(gram.data.tobytes())
        dump(gram, os.path.join(config["TMP"], f"gram{h}"), compress=3)
        del partition_descriptor, gram, m1, m2, kernel
        _ = gc.collect()
        return h

    combinations = list(combinations_with_replacement(enumerate(features), 2))
    chunked_combs = chunks(combinations, int(len(combinations) / config["N_CPUS"]))
    gram_hashes = Parallel(n_jobs=min(config["N_CPUS"], 10), verbose=0)(
        delayed(calc_gram_row)(combs) for combs in chunked_combs
    )
    return np.sum(
        [load(os.path.join(config["TMP"], f"gram{h}")) for h in gram_hashes],
        axis=0,
    )


def rf_kernel(rfc, features, n_samples=500):
    tree_indices = np.random.randint(low=0, high=len(rfc.estimators_), size=n_samples)
    forest = rfc.estimators_
    parallel_node_ids(forest, tree_indices, n_samples)
    parallel_partition_desc(forest=forest, tree_indices=tree_indices, features=features)
    gram = parallel_gram(features, n_samples)
    return gram
