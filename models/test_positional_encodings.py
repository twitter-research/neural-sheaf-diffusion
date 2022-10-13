# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
import torch

from torch import tensor
from models.positional_encodings import append_top_k_evectors, get_laplacian_evectors
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import csr_matrix
from utils.heterophilic import get_dataset


class PosEncTests(unittest.TestCase):
    def setUp(self):
        self.edge_index = tensor([[0, 2, 2, 1], [1, 0, 1, 2]]).t()
        self.edge_weight = torch.ones(self.edge_index.size(0), dtype=torch.long)
        self.test_edges = tensor([[0, 1], [1, 2]]).t()
        self.num_nodes = 3

    def tearDown(self) -> None:
        pass

    def test_evector(self):
        class test_dataset:
            def __init__(self, data):
                self.data = data
                self.name = 'TestData'

        class test_data:
            def __init__(self, edge_index, x):
                self.edge_index = edge_index
                self.x = x
                self.num_nodes = x.shape[0]

        k = 1
        edge_index = torch.tensor([[0, 0, 1, 2, 2, 1], [1, 2, 0, 0, 1, 2]])
        x = torch.ones((3, 2))
        dataset = test_dataset(test_data(edge_index, x))
        dataset = append_top_k_evectors(dataset, k=k)
        self.assertTrue(dataset.data.x.shape == (3, 2 + k))

    def test_get_laplacian_evectors_on_texas(self):
        dataset = get_dataset('texas')
        A = to_scipy_sparse_matrix(dataset.data.edge_index)
        diag = csr_matrix.diagonal(A.tocsr())
        self.assertTrue(diag.sum() == 0)
        # get the smallest k+1 eigenpairs and then throw away the smallest, which is constant
        k = 2
        eigenvalues, eigenvectors = get_laplacian_evectors(A, k + 1)
        norm = np.linalg.norm(eigenvectors[:, 1])
        self.assertAlmostEqual(norm, 1, places=4)
        self.assertAlmostEqual(eigenvalues[0], 0, places=4)

    def test_append_evector_on_texas(self):
        dataset = get_dataset('texas')
        n_nodes, n_features = dataset.data.x.shape
        k = 0
        dataset = append_top_k_evectors(dataset, k=k)
        self.assertTrue(dataset.data.x.shape == (n_nodes, n_features + k))
        self.assertTrue(n_features == dataset.data.x.shape[1])
        k = 2
        dataset = append_top_k_evectors(dataset, k=k)
        self.assertTrue(dataset.data.x.shape == (n_nodes, n_features + k))
