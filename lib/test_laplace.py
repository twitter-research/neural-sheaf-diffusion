# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import numpy as np
import networkx as nx

from scipy import linalg
from lib.laplace import build_sheaf_laplacian, build_norm_sheaf_laplacian, build_sheaf_difussion_matrix, dirichlet_energy


def build_dense_laplacian(graph, K, normalised=False, augmented=False):
    """Builds a sheaf laplacian from a given graph using naive computations."""
    B_T = nx.incidence_matrix(graph, oriented=True).toarray().T
    E, N = B_T.shape

    # Build unnormalised Laplacian.
    one_matrix = np.ones((K, K))
    B_T_block = np.kron(B_T, one_matrix)
    Delta_base = np.random.uniform(low=-1.0, high=1.0, size=(E * K, N * K))
    Delta = B_T_block * Delta_base
    L_dense = Delta.T @ Delta

    if not normalised:
        return torch.tensor(Delta), torch.tensor(L_dense, dtype=torch.float)

    # Build normalised Laplacian.
    I = np.identity(N)
    D_ones = np.kron(I, one_matrix)
    D = L_dense * D_ones
    if augmented:
        D = D + np.identity(N*K)
    D_inv = linalg.fractional_matrix_power(D, -0.5)
    nL_dense = D_inv @ L_dense @ D_inv

    return torch.tensor(Delta), torch.tensor(nL_dense, dtype=torch.float)


def build_restriction_maps(graph, K, Delta):
    """Constructs the restriction maps of a sheaf from the sheaf co-boundary operator."""
    B_T = nx.incidence_matrix(graph, oriented=True).toarray().T
    E, N = B_T.shape

    # Build a sparse sheaf Laplacian
    edge_index = torch.empty((2, E), dtype=torch.long)
    maps = torch.empty((E, 2, K, K), dtype=torch.float)

    for e in range(E):
        source = None
        target = None
        for n in range(N):
            if B_T[e, n] == -1:
                source = n
                maps[e, 0, :, :] = -Delta[e * K:(e + 1) * K, n * K:(n + 1) * K].clone().detach()
            elif B_T[e, n] == 1:
                target = n
                maps[e, 1, :, :] = Delta[e * K:(e + 1) * K, n * K:(n + 1) * K].clone().detach()

        edge_index[0, e] = source
        edge_index[1, e] = target

    return edge_index, maps


@pytest.mark.parametrize("graph_id,K", [(3, 5), (1003, 10), (1241, 1)])
def test_build_sheaf_laplacian(graph_id, K):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Use a simple graph from the graph atlas.
    graph = nx.graph_atlas(graph_id)
    N = len(graph)

    # Build a dense sheaf Laplacian
    Delta, L_dense = build_dense_laplacian(graph, K)

    # Generate an edge_index object and the associated restriction maps
    edge_index, maps = build_restriction_maps(graph, K, Delta)

    # Compute the sparse sheaf Laplacian efficiently
    index, values = build_sheaf_laplacian(N, K, edge_index, maps)
    L = torch.sparse_coo_tensor(index, values, size=(N*K, N*K)).to_dense()

    assert torch.all(torch.abs(L - L_dense) < 1e-5)


@pytest.mark.parametrize("graph_id,K,augmented", [(3, 5, False), (1003, 10, True), (1241, 1, True)])
def test_build_norm_sheaf_laplacian(graph_id, K, augmented):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Use a simple graph from the graph atlas.
    graph = nx.graph_atlas(graph_id)
    N = len(graph)

    # Build a dense normalised sheaf Laplacian
    Delta, nL_dense = build_dense_laplacian(graph, K, normalised=True, augmented=augmented)

    # Generate an edge_index object and the associated restriction maps
    edge_index, maps = build_restriction_maps(graph, K, Delta)

    index, values = build_norm_sheaf_laplacian(N, K, edge_index, maps, augmented=augmented)
    nL = torch.sparse_coo_tensor(index, values, size=(N * K, N * K)).to_dense()

    assert torch.all(torch.abs(nL - nL_dense) < 1e-5)


@pytest.mark.parametrize("graph_id,K,augmented", [(3, 5, False), (1003, 10, True), (1241, 1, True)])
def test_build_sheaf_difussion_map(graph_id, K, augmented):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Use a simple graph from the graph atlas.
    graph = nx.graph_atlas(graph_id)
    N = len(graph)

    # Build a dense normalised sheaf Laplacian
    Delta, nL_dense = build_dense_laplacian(graph, K, normalised=True, augmented=augmented)
    P_dense = torch.eye(N*K, N*K) - nL_dense

    # Generate an edge_index object and the associated restriction maps
    edge_index, maps = build_restriction_maps(graph, K, Delta)

    index, values = build_sheaf_difussion_matrix(N, K, edge_index, maps, augmented=augmented)
    P = torch.sparse_coo_tensor(index, values, size=(N * K, N * K)).to_dense()

    assert torch.all(torch.abs(P - P_dense) < 1e-5)


@pytest.mark.parametrize("graph_id,K,augmented", [(3, 5, False), (1003, 10, True), (1241, 1, True)])
def test_dirichlet_energy(graph_id, K, augmented):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Use a simple graph from the graph atlas.
    graph = nx.graph_atlas(graph_id)
    N = len(graph)

    # Build a dense normalised sheaf Laplacian
    Delta, nL_dense = build_dense_laplacian(graph, K, normalised=True, augmented=augmented)

    # Generate an edge_index object and the associated restriction maps
    edge_index, maps = build_restriction_maps(graph, K, Delta)

    nL = build_norm_sheaf_laplacian(N, K, edge_index, maps, augmented=augmented)

    # Compute the dirichlet energy
    f = torch.FloatTensor(size=(N*K, 1)).uniform_(-1.0, 1.0)
    energy_from_sparse = dirichlet_energy(nL, f, size=N*K)
    energy_from_dense = f.t() @ nL_dense @ f

    assert energy_from_sparse > -1e-10
    assert abs(energy_from_dense - energy_from_sparse) < 1e-5










