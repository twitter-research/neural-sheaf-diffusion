# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Add the top k eigenvectors to the node features of a pyg dataset
"""
import os
import torch
import numpy as np
import pickle

from scipy.sparse import linalg, diags
from torch_geometric.utils import to_scipy_sparse_matrix
from definitions import ROOT_DIR

POS_ENC_PATH = os.path.join(ROOT_DIR, "datasets", "pos_encodings")


def find_or_make_encodings(dataset, k, max_evector_dim=64):
    # generate new positional encodings
    # do encodings already exist on disk?
    # there are only num_nodes evectors, and the evector lib fails when k=num_nodes
    if dataset.data.num_nodes < max_evector_dim:
        max_evector_dim = dataset.data.num_nodes - 2
    assert k <= max_evector_dim, f'maximum number of eigenvectors to cache {max_evector_dim} is less than the number requested {k}'

    fname = os.path.join(POS_ENC_PATH, f"{dataset.name}_evectors_{max_evector_dim}.pkl")
    print(f"[i] Looking for positional encodings in {fname}...")

    # - if so, just load them
    if os.path.exists(fname):
        print("    Found them! Loading cached version")
        with open(fname, "rb") as f:
            eigenvectors = pickle.load(f)

    # - otherwise, calculate...
    else:
        print("Encodings not found! Calculating and caching them...")
        # choose different functions for different positional encodings
        data = dataset.data
        # todo this should check if they already exist on disk and only generate if they don't
        A = to_scipy_sparse_matrix(data.edge_index)
        # get the smallest k+1 eigenpairs and then throw away the smallest, which is constant
        _, all_eigenvectors = get_laplacian_evectors(A, max_evector_dim + 1)
        eigenvectors = all_eigenvectors[:, 1:]
        # - ... and store them on disk
        if not os.path.exists(POS_ENC_PATH):
            os.makedirs(POS_ENC_PATH)
        with open(fname, "wb") as f:
            pickle.dump(eigenvectors, f)

    return eigenvectors


def append_top_k_evectors(dataset, k=2):
    eigenvectors = find_or_make_encodings(dataset, k, max_evector_dim=64)
    data = dataset.data
    # # todo this should check if they already exist on disk and only generate if they don't
    # A = to_scipy_sparse_matrix(data.edge_index)
    # # get the smallest k+1 eigenpairs and then throw away the smallest, which is constant
    # _, eigenvectors = get_laplacian_evectors(A, k + 1)
    data.x = torch.cat([data.x, torch.from_numpy(eigenvectors[:, :k])], dim=1)
    dataset.data = data
    return dataset


def get_laplacian_evectors(A, k):
    degree_mat = diags(np.squeeze(np.asarray(A.sum(axis=1))))
    L = degree_mat - A
    eigenvalues, eigenvectors = linalg.eigsh(L, which='SA', k=k)
    return eigenvalues, eigenvectors


if __name__ == '__main__':
    class test_dataset:
        def __init__(self, edge_index, x):
            self.edge_index = edge_index
            self.x = x

    edge_index = torch.tensor([[0, 2, 2, 1], [1, 0, 1, 2]])
    edge_weight = torch.ones(edge_index.size(0), dtype=int)
    x = torch.ones((3, 2))
    dataset = test_dataset(edge_index, x)
    dataset = append_top_k_evectors(dataset, k=2)
