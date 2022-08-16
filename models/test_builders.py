# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import numpy as np
import networkx as nx
import lib.laplace as lap

from models.laplacian_builders import DiagLaplacianBuilder, GeneralLaplacianBuilder, NormConnectionLaplacianBuilder
from torch_geometric.utils import from_networkx
from lib.laplace import build_dense_laplacian
from utils.heterophilic import get_dataset


@pytest.mark.parametrize("graph_id, d, normalised",
                         [(3, 1, False), (50, 5, False), (1005, 7, False), (10, 1, False),
                          (10, 1, True), (70, 3, True), (650, 11, True)])
def test_diag_laplacian_builder_produces_valid_laplacian(graph_id, d, normalised):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    graph = nx.graph_atlas(graph_id)
    size = len(graph)
    data = from_networkx(graph)

    # Construct the model
    model = DiagLaplacianBuilder(size, data.edge_index, d=d, normalised=normalised)
    maps = torch.FloatTensor(size=(data.edge_index.size(1), d)).uniform_(-1.0, 1.0)
    L, _ = model(maps)
    assert torch.isfinite(L[1]).all()

    L = torch.sparse_coo_tensor(L[0], L[1], size=(size*d, size*d), dtype=torch.float64).to_dense()
    Q, V = torch.linalg.eigh(L)

    # Check that L is semi-positive definite.
    eps = 1e-6
    assert torch.equal(L, L.T)
    assert torch.all(Q >= 0 - eps)
    if normalised:
        assert torch.all(Q <= 2. + eps)

    L_dense = build_dense_laplacian(size, data.edge_index, maps.to(dtype=torch.float64), d=d,
                                    normalised=normalised, diagonal_maps=True)
    assert torch.allclose(L, L_dense, atol=eps)


@pytest.mark.parametrize("graph_id, d, normalised",
                         [(3, 1, False), (50, 5, False), (1005, 7, False), (10, 1, False),
                          (10, 1, True), (70, 3, True), (650, 11, True)])
def test_diag_laplacian_builder_with_fixed_maps_produces_valid_laplacian(graph_id, d, normalised):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    graph = nx.graph_atlas(graph_id)
    size = len(graph)
    data = from_networkx(graph)

    # Construct the model
    model = DiagLaplacianBuilder(size, data.edge_index, d=d, normalised=normalised, add_hp=True, add_lp=True)
    maps = torch.FloatTensor(size=(data.edge_index.size(1), d)).uniform_(-1.0, 1.0)
    L, _ = model(maps)
    assert torch.isfinite(L[1]).all()

    L = torch.sparse_coo_tensor(L[0], L[1], size=(size*(d+2), size*(d+2)), dtype=torch.float64).to_dense()
    Q, V = torch.linalg.eigh(L)

    # Check that L is semi-positive definite.
    eps = 1e-6
    assert torch.equal(L, L.T)
    assert torch.all(Q >= 0 - eps)
    if normalised:
        assert torch.all(Q <= 2. + eps)

    # Append fixed submaps to the restriction maps.
    values = [1.0, -1.0]
    L_dense = build_dense_laplacian(size, data.edge_index, maps.to(dtype=torch.float64), d=d,
                                    normalised=normalised, diagonal_maps=True, values=values)
    assert torch.allclose(L, L_dense, atol=eps)


@pytest.mark.parametrize("graph_id,d,normalised,augmented",
                         [(3, 2, False, False), (3, 1, False, False), (1150, 2, False, False),
                          (10, 7, True, False), (56, 9, True, False), (876, 5, True, True), (70, 3, True, True)])
def test_general_laplacian_builder_produces_valid_laplacian(graph_id, d, normalised, augmented):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    graph = nx.graph_atlas(graph_id)
    size = len(graph)
    data = from_networkx(graph)

    # Construct the model
    model = GeneralLaplacianBuilder(size, data.edge_index, d, normalised=normalised)
    model.eval()
    maps = torch.FloatTensor(size=(data.edge_index.size(1), d, d)).uniform_(-1.0, 1.0)
    L, _ = model(maps)
    L = torch.sparse_coo_tensor(L[0], L[1], size=(size*d, size*d), dtype=torch.float64).to_dense()
    Q, _ = torch.linalg.eigh(L)

    # Check that L is semi-positive definite.
    eps = 1e-6
    assert torch.allclose(L, L.T, atol=eps)
    assert torch.all(Q >= 0 - eps), Q
    if normalised:
        assert torch.all(Q <= 2.)

    L_dense = build_dense_laplacian(size, data.edge_index, maps.to(dtype=torch.float64), d, normalised=normalised)
    assert torch.allclose(L, L_dense, atol=eps)


@pytest.mark.parametrize("graph_id, d",
                         [(3, 2), (3, 1), (1150, 2), (10, 7), (56, 9), (876, 5), (70, 3)])
def test_general_laplacian_builder_with_deg_normalisation_produces_valid_laplacian(graph_id, d):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    graph = nx.graph_atlas(graph_id)
    size = len(graph)
    data = from_networkx(graph)

    # Construct the model
    model = GeneralLaplacianBuilder(size, data.edge_index, d, deg_normalised=True)
    model.eval()

    maps = torch.FloatTensor(size=(data.edge_index.size(1), d, d)).uniform_(-1.0, 1.0)
    L, _ = model(maps)
    L = torch.sparse_coo_tensor(L[0], L[1], size=(size*d, size*d), dtype=torch.float64).to_dense()
    Q, _ = torch.linalg.eigh(L)

    # Check that L is semi-positive definite.
    eps = 1e-6
    assert torch.allclose(L, L.T, atol=eps)
    assert torch.all(Q >= 0 - eps), Q
    assert torch.all(Q <= 2.)


@pytest.mark.parametrize("graph_id,d,normalised,augmented",
                         [(3, 2, False, False), (3, 1, False, False), (1150, 2, False, False),
                          (10, 7, True, False), (56, 9, True, False), (876, 5, True, True), (70, 3, True, True)])
def test_general_laplacian_builder_with_fixed_maps_produces_valid_laplacian(graph_id, d, normalised, augmented):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    graph = nx.graph_atlas(graph_id)
    size = len(graph)
    data = from_networkx(graph)

    # Construct the model
    model = GeneralLaplacianBuilder(size, data.edge_index, d, normalised=normalised, add_hp=True, add_lp=True)
    model.eval()

    maps = torch.FloatTensor(size=(data.edge_index.size(1), d, d)).uniform_(-1.0, 1.0)
    L, _ = model(maps)
    L = torch.sparse_coo_tensor(L[0], L[1], size=(size*(d+2), size*(d+2)), dtype=torch.float64).to_dense()
    Q, _ = torch.linalg.eigh(L)

    # Check that L is semi-positive definite.
    eps = 1e-6
    assert torch.allclose(L, L.T, atol=eps)
    assert torch.all(Q >= 0 - eps), Q
    if normalised:
        assert torch.all(Q <= 2.)

    values = [1.0, -1.0]
    L_dense = build_dense_laplacian(size, data.edge_index, maps.to(dtype=torch.float64),
                                    d=d, normalised=normalised, values=values)
    assert torch.allclose(L, L_dense, atol=eps)


@pytest.mark.parametrize("d,normalised,augmented",
                         [(2, False, False), (1, False, False), (2, False, True),
                          (7, True, False), (9, True, False), (5, True, True), (3, True, True)])
def test_general_laplacian_builder_produces_valid_laplacian_on_texas(d, normalised, augmented):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    data = get_dataset('texas')[0]
    size = data.x.size(0)

    # Construct the model
    model = GeneralLaplacianBuilder(size, data.edge_index, d, normalised=normalised)
    model.eval()

    maps = torch.FloatTensor(size=(data.edge_index.size(1), d, d)).uniform_(-1.0, 1.0)
    L, _ = model(maps)
    L = torch.sparse_coo_tensor(L[0], L[1], size=(size*d, size*d), dtype=torch.float64).to_dense()
    Q, _ = torch.linalg.eigh(L)

    # Check that L is semi-positive definite.
    eps = 1e-5
    assert torch.allclose(L, L.T, atol=eps)
    assert torch.all(Q >= 0 - eps), Q
    if normalised:
        assert torch.all(Q <= 2.)

    L_dense = build_dense_laplacian(size, data.edge_index, maps, d, normalised=normalised)
    assert torch.allclose(L, L_dense, atol=eps)


@pytest.mark.parametrize("graph_id, d, orth_trans",
                         [(3, 2, "euler"), (3, 3, "euler"), (1150, 2, "householder"), (10, 7, "householder"),
                          (56, 9, "matrix_exp"), (876, 5, "matrix_exp"), (70, 3, "cayley")])
def test_norm_connection_laplacian_produces_valid_laplacian(graph_id, d, orth_trans):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    graph = nx.graph_atlas(graph_id)
    size = len(graph)
    data = from_networkx(graph)

    # Construct the model
    model = NormConnectionLaplacianBuilder(size, edge_index=data.edge_index, d=d, orth_map=orth_trans)
    # Generate maps
    samples = data.edge_index.size(1)
    if orth_trans in ["householder", "euler"]:
        map_params = torch.FloatTensor(size=(samples, d * (d - 1) // 2)).uniform_(-1.0, 1.0)
    else:
        map_params = torch.FloatTensor(size=(samples, d * (d + 1) // 2)).uniform_(-1.0, 1.0)
    maps = model.orth_transform(map_params)

    L, _ = model(map_params)
    L = torch.sparse_coo_tensor(L[0], L[1], size=(size*d, size*d), dtype=torch.float64).to_dense()
    Q, _ = torch.linalg.eigh(L)

    # Check that L is semi-positive definite.
    eps = 1e-6
    assert torch.equal(L, L.T)
    assert torch.all(Q >= 0 - eps), Q
    assert torch.all(Q <= 2. + eps), Q

    L_dense = build_dense_laplacian(size, data.edge_index, maps, d, normalised=True)
    assert torch.allclose(L, L_dense, atol=eps)


@pytest.mark.parametrize("graph_id, d, orth_trans",
                         [(3, 2, "euler"), (3, 3, "euler"), (1150, 2, "householder"), (10, 7, "householder"),
                          (56, 9, "matrix_exp"), (876, 5, "matrix_exp"), (70, 3, "cayley")])
def test_norm_connection_laplacian_with_fixed_maps_produces_valid_laplacian(graph_id, d, orth_trans):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    graph = nx.graph_atlas(graph_id)
    size = len(graph)
    data = from_networkx(graph)

    # Construct the model
    model = NormConnectionLaplacianBuilder(
        size, edge_index=data.edge_index, d=d, orth_map=orth_trans, add_lp=True, add_hp=True)
    # Generate maps
    samples = data.edge_index.size(1)
    if orth_trans in ["householder", "euler"]:
        map_params = torch.FloatTensor(size=(samples, d * (d - 1) // 2)).uniform_(-1.0, 1.0)
    else:
        map_params = torch.FloatTensor(size=(samples, d * (d + 1) // 2)).uniform_(-1.0, 1.0)
    maps = model.orth_transform(map_params)

    L, _ = model(map_params)
    L = torch.sparse_coo_tensor(L[0], L[1], size=(size*(d+2), size*(d+2)), dtype=torch.float64).to_dense()
    Q, _ = torch.linalg.eigh(L)

    # Check that L is semi-positive definite.
    eps = 1e-6
    assert torch.equal(L, L.T)
    assert torch.all(Q >= 0 - eps), Q
    assert torch.all(Q <= 2. + eps), Q

    # Append fixed submaps to the restriction maps.
    values = [1.0, -1.0]
    L_dense = build_dense_laplacian(size, data.edge_index, maps, d, normalised=True, values=values)
    assert torch.allclose(L, L_dense, atol=eps)


@pytest.mark.parametrize("graph_id, d, orth_trans",
                         [(3, 2, "euler"), (3, 3, "euler"), (1150, 2, "householder"), (10, 7, "householder"),
                          (56, 9, "matrix_exp"), (876, 5, "matrix_exp"), (70, 3, "cayley")])
def test_norm_connection_laplacian_with_edge_weights_produces_valid_laplacian(graph_id, d, orth_trans):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    graph = nx.graph_atlas(graph_id)
    size = len(graph)
    data = from_networkx(graph)

    # Construct the model
    model = NormConnectionLaplacianBuilder(
        size, edge_index=data.edge_index, d=d, orth_map=orth_trans, add_lp=True, add_hp=True)
    # Generate maps
    samples = data.edge_index.size(1)
    if orth_trans in ["householder", "euler"]:
        map_params = torch.FloatTensor(size=(samples, d * (d - 1) // 2)).uniform_(-1.0, 1.0)
    else:
        map_params = torch.FloatTensor(size=(samples, d * (d + 1) // 2)).uniform_(-1.0, 1.0)

    edge_weights = lap.get_random_edge_weights(data.edge_index)
    # Use edge weights
    maps = model.orth_transform(map_params)

    L, _ = model(map_params, edge_weights)
    L = torch.sparse_coo_tensor(L[0], L[1], size=(size*(d+2), size*(d+2)), dtype=torch.float64).to_dense()
    Q, _ = torch.linalg.eigh(L)

    # Check that L is semi-positive definite.
    eps = 1e-6
    assert torch.equal(L, L.T)
    assert torch.all(Q >= 0 - eps), Q
    assert torch.all(Q <= 2. + eps), Q

    # Append fixed submaps to the restriction maps.
    values = [1.0, -1.0]
    L_dense = build_dense_laplacian(size, data.edge_index, maps, d, normalised=True, values=values,
                                    edge_weights=edge_weights)
    assert torch.allclose(L, L_dense, atol=eps)