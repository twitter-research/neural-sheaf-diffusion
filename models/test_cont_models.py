# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import numpy as np
import networkx as nx
import torch.nn.functional as F

from nsd.lib.perm_utils import generate_permutation_matrices, permute_graph
from nsd.utils.heterophilic import get_dataset
from nsd.models.cont_models import (
    GraphLaplacianDiffusion,
    DiagSheafDiffusion,
    GeneralSheafDiffusion,
    ODEBlock,
    LaplacianODEFunc,
    BundleSheafDiffusion,
)
from torch_geometric.utils import from_networkx, get_laplacian, degree


def get_test_config():
    return {
        "hidden_channels": 5,
        "device": torch.device("cpu"),
        "layers": 2,
        "normalised": False,
        "deg_normalised": False,
        "linear": False,
        "input_dropout": 0.0,
        "dropout": 0.0,
        "left_weights": True,
        "right_weights": True,
        "use_act": True,
        "second_linear": True,
        "add_lp": True,
        "add_hp": True,
        "max_t": 1.0,
        "sheaf_act": "tanh",
        "tol_scale": 1.0,
        "int_method": "euler",
        "step_size": 0.1,
        "max_iters": 100,
        "tol_scale_adjoint": 1.0,
        "adjoint_method": "adaptive_heun",
        "adjoint_step_size": 0.1,
        "edge_weights": False,
        "orth": "householder",
        "sparse_learner": False,
    }


@pytest.mark.parametrize("adjoint", [False, True])
def test_permutation_equivariance_of_graph_laplacian_difussion(adjoint):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    dataset = get_dataset("texas")
    data = dataset[0]

    perms = 3
    Ps = generate_permutation_matrices(size=data.x.size(0), amount=perms)

    args = get_test_config()
    args["graph_size"] = data.size(0)
    args["input_dim"] = dataset.num_features
    args["output_dim"] = dataset.num_classes
    args["d"] = 1
    args["normalised"] = True
    args["deg_normalised"] = False
    args["adjoint"] = adjoint

    # Construct the model
    for i in range(perms):
        with torch.no_grad():
            # init model
            model = GraphLaplacianDiffusion(data.edge_index, args)
            model.eval()

            # run diffusion
            out = model(data.x)

            # apply perm matrix
            out = torch.FloatTensor(Ps[i] @ out.numpy())

            # permute the graph; update edge_index in model; run diffusion again
            perm_data = permute_graph(data.clone(), Ps[i])
            model.update_edge_index(perm_data.edge_index)
            perm_out = model(perm_data.x)

        assert torch.allclose(out, perm_out, atol=1e-6)


@pytest.mark.parametrize(
    "graph_id, linear, adjoint", [(304, False, False), (3, True, True), (1009, False, True)]
)
def test_backprop_in_diag_sheaf_difussion(graph_id, linear, adjoint):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    num_feat = 13
    graph = nx.graph_atlas(graph_id)
    data = from_networkx(graph)
    data.x = torch.FloatTensor(size=(len(graph), num_feat)).uniform_(-1.0, 1.0)
    data.y = torch.tensor([1] * len(graph), dtype=torch.long)

    # Get args
    args = get_test_config()
    args["graph_size"] = data.size(0)
    args["input_dim"] = num_feat
    args["output_dim"] = 2
    args["d"] = 1
    args["normalised"] = True
    args["deg_normalised"] = False
    args["linear"] = linear
    args["adjoint"] = adjoint

    # Construct the model
    model = GraphLaplacianDiffusion(data.edge_index, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for _ in range(2):
        optimizer.zero_grad()
        out = model(data.x)
        assert list(out.size()) == [len(graph), 2]

        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

        # Check that gradients are back-propagated through the Laplacian
        for param in model.parameters():
            assert param.grad is not None


@pytest.mark.parametrize("d, adjoint", [(3, False), (5, True), (7, False), (7, True)])
def test_permutation_equivariance_of_diag_sheaf_diffusion_on_texas(d, adjoint):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    dataset = get_dataset("texas")
    data = dataset[0]

    perms = 3
    Ps = generate_permutation_matrices(size=data.x.size(0), amount=perms)

    args = get_test_config()
    args["graph_size"] = data.size(0)
    args["input_dim"] = dataset.num_features
    args["output_dim"] = dataset.num_classes
    args["d"] = d
    args["normalised"] = True
    args["deg_normalised"] = False
    args["adjoint"] = adjoint

    # Construct the model
    for i in range(perms):
        with torch.no_grad():
            model = DiagSheafDiffusion(data.edge_index, args)
            model.eval()
            out = model(data.x)
            out = torch.FloatTensor(Ps[i] @ out.numpy())

            perm_data = permute_graph(data.clone(), Ps[i])
            model.update_edge_index(perm_data.edge_index)
            perm_out = model(perm_data.x)

        assert torch.allclose(out, perm_out, atol=1e-6)


@pytest.mark.parametrize(
    "graph_id, d, linear, adjoint",
    [(304, 1, False, False), (3, 5, True, True), (1009, 7, False, True)],
)
def test_backprop_in_diag_sheaf_difussion(graph_id, d, linear, adjoint):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    num_feat = 13
    graph = nx.graph_atlas(graph_id)
    data = from_networkx(graph)
    data.x = torch.FloatTensor(size=(len(graph), num_feat)).uniform_(-1.0, 1.0)
    data.y = torch.tensor([1] * len(graph), dtype=torch.long)

    # Get args
    args = get_test_config()
    args["graph_size"] = data.size(0)
    args["input_dim"] = num_feat
    args["output_dim"] = 2
    args["d"] = d
    args["normalised"] = True
    args["deg_normalised"] = False
    args["linear"] = linear
    args["adjoint"] = adjoint

    # Construct the model
    model = DiagSheafDiffusion(data.edge_index, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for _ in range(2):
        optimizer.zero_grad()
        out = model(data.x)
        assert list(out.size()) == [len(graph), 2]

        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

        # Check that gradients are back-propagated through the Laplacian
        for param in model.parameters():
            assert param.grad is not None


@pytest.mark.parametrize("d, adjoint", [(3, False), (5, True), (7, False), (7, True)])
def test_permutation_equivariance_of_bundle_sheaf_diffusion_on_texas(d, adjoint):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    dataset = get_dataset("texas")
    data = dataset[0]

    perms = 3
    Ps = generate_permutation_matrices(size=data.x.size(0), amount=perms)

    args = get_test_config()
    args["graph_size"] = data.size(0)
    args["input_dim"] = dataset.num_features
    args["output_dim"] = dataset.num_classes
    args["d"] = d
    args["normalised"] = True
    args["deg_normalised"] = False
    args["adjoint"] = adjoint
    args["edge_weights"] = True

    # Construct the model
    for i in range(perms):
        with torch.no_grad():
            model = BundleSheafDiffusion(data.edge_index, args)
            model.eval()
            out = model(data.x)
            out = torch.FloatTensor(Ps[i] @ out.numpy())

            perm_data = permute_graph(data.clone(), Ps[i])
            model.update_edge_index(perm_data.edge_index)
            perm_out = model(perm_data.x)

        assert torch.allclose(out, perm_out, atol=1e-6)


@pytest.mark.parametrize(
    "graph_id, d, linear, adjoint",
    [(304, 2, False, False), (3, 5, True, True), (1009, 7, False, True)],
)
def test_backprop_in_bundle_sheaf_difussion(graph_id, d, linear, adjoint):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    num_feat = 13
    graph = nx.graph_atlas(graph_id)
    data = from_networkx(graph)
    data.x = torch.FloatTensor(size=(len(graph), num_feat)).uniform_(-1.0, 1.0)
    data.y = torch.tensor([1] * len(graph), dtype=torch.long)

    # Get args
    args = get_test_config()
    args["graph_size"] = data.size(0)
    args["input_dim"] = num_feat
    args["output_dim"] = 2
    args["d"] = d
    args["normalised"] = True
    args["deg_normalised"] = False
    args["linear"] = linear
    args["adjoint"] = adjoint
    args["edge_weights"] = True

    # Construct the model
    model = BundleSheafDiffusion(data.edge_index, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for _ in range(2):
        optimizer.zero_grad()
        out = model(data.x)
        assert list(out.size()) == [len(graph), 2]

        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

        # Check that gradients are back-propagated through the Laplacian
        for param in model.parameters():
            assert param.grad is not None


@pytest.mark.parametrize("d, adjoint", [(3, False), (5, True), (7, False), (7, True)])
def test_permutation_equivariance_of_general_sheaf_diffusion_on_texas(d, adjoint):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    dataset = get_dataset("texas")
    data = dataset[0]

    perms = 3
    Ps = generate_permutation_matrices(size=data.x.size(0), amount=perms)

    args = get_test_config()
    args["graph_size"] = data.size(0)
    args["input_dim"] = dataset.num_features
    args["output_dim"] = dataset.num_classes
    args["d"] = d
    args["normalised"] = True
    args["deg_normalised"] = False
    args["adjoint"] = adjoint

    # Construct the model
    for i in range(perms):
        with torch.no_grad():
            model = GeneralSheafDiffusion(data.edge_index, args)
            model.eval()
            out = model(data.x)
            out = torch.FloatTensor(Ps[i] @ out.numpy())

            perm_data = permute_graph(data.clone(), Ps[i])
            model.update_edge_index(perm_data.edge_index)
            perm_out = model(perm_data.x)

        assert torch.allclose(out, perm_out, atol=1e-6)


@pytest.mark.parametrize("graph_id, d, adjoint", [(304, 2, True), (3, 5, False), (1009, 7, False)])
def test_backprop_in_general_sheaf_difussion(graph_id, d, adjoint):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    num_feat = 13
    graph = nx.graph_atlas(graph_id)
    data = from_networkx(graph)
    data.x = torch.FloatTensor(size=(len(graph), num_feat)).uniform_(-1.0, 1.0)
    data.y = torch.tensor([1] * len(graph), dtype=torch.long)

    # Get args
    args = get_test_config()
    args["graph_size"] = data.size(0)
    args["input_dim"] = num_feat
    args["output_dim"] = 2
    args["d"] = d
    args["normalised"] = True
    args["deg_normalised"] = False
    args["adjoint"] = adjoint

    # Construct the model
    model = GeneralSheafDiffusion(data.edge_index, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    optimizer.zero_grad()
    out = model(data.x)
    assert list(out.size()) == [len(graph), 2]

    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()

    # Check that gradients are back-propagated through the Laplacian
    for param in model.parameters():
        assert param.grad is not None
