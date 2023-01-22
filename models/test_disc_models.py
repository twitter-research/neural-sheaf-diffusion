# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import numpy as np
import networkx as nx
import torch.nn.functional as F

from lib.perm_utils import generate_permutation_matrices, permute_graph
from models.disc_models import DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion, DiscreteGeneralSheafDiffusion
from torch_geometric.utils import from_networkx
from utils.heterophilic import get_dataset


def get_test_config():
    return {
        'hidden_channels': 5,
        'device': torch.device('cpu'),
        'layers': 2,
        'normalised': False,
        'deg_normalised': False,
        'linear': False,
        'input_dropout': 0.5,
        'dropout': 0.5,
        'left_weights': True,
        'right_weights': True,
        'use_act': True,
        'second_linear': True,
        'add_lp': True,
        'add_hp': True,
        'max_t': 1.0,
        'sheaf_act': 'tanh',
        'tol_scale': 1.0,
        'int_method': 'euler',
        'step_size': 0.1,
        'max_iters': 100,
        'tol_scale_adjoint': 1.0,
        'adjoint_method': 'adaptive_heun',
        'adjoint_step_size': 0.1,
        'edge_weights': True,
        'orth': 'householder',
        'sparse_learner': False,
    }


@pytest.mark.parametrize("graph_id, d", [(256, 7), (70, 5), (379, 3), (1200, 5)])
def test_permutation_equivariance_of_diag_sheaf_diffusion_on_graph_atals(graph_id, d):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    graph = nx.graph_atlas(graph_id)
    data = from_networkx(graph)
    data.x = torch.FloatTensor(size=(len(graph), 13)).uniform_(-1.0, 1.0)

    args = get_test_config()
    args['graph_size'] = data.size(0)
    args['input_dim'] = 13
    args['output_dim'] = 3
    args['d'] = d
    args['normalised'] = True
    args['deg_normalised'] = False

    P = generate_permutation_matrices(size=len(graph), amount=1)[0]
    perm_data = permute_graph(data, P)

    # Construct the model
    with torch.no_grad():
        model = DiscreteDiagSheafDiffusion(data.edge_index, args)
        model.eval()
        out = model(data.x)
        out = torch.FloatTensor(P.astype(np.float64) @ out.numpy().astype(np.float64))

        model.update_edge_index(perm_data.edge_index)
        perm_out = model(perm_data.x)

    assert torch.allclose(out, perm_out, atol=1e-6)


@pytest.mark.parametrize("d, normalised", [(3, True), (5, False), (7, True)])
def test_permutation_equivariance_of_diag_sheaf_diffusion_on_texas(d, normalised):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    dataset = get_dataset('texas')
    data = dataset[0]

    perms = 3
    Ps = generate_permutation_matrices(size=data.x.size(0), amount=perms)

    args = get_test_config()
    args['graph_size'] = data.size(0)
    args['input_dim'] = dataset.num_features
    args['output_dim'] = dataset.num_classes
    args['d'] = d
    args['normalised'] = normalised
    args['deg_normalised'] = not normalised

    # Construct the model
    for i in range(perms):
        with torch.no_grad():
            model = DiscreteDiagSheafDiffusion(data.edge_index, args)
            model.eval()
            out = model(data.x)
            out = torch.FloatTensor(Ps[i] @ out.numpy())

            perm_data = permute_graph(data.clone(), Ps[i])
            model.update_edge_index(perm_data.edge_index)
            perm_out = model(perm_data.x)

        assert torch.allclose(out, perm_out, atol=1e-5)


@pytest.mark.parametrize("d, orth", [(2, "euler"), (3, "euler"), (5, "matrix_exp"), (6, "cayley"), (7, "householder")])
def test_permutation_equivariance_of_bundle_sheaf_diffusion_on_texas(d, orth):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    dataset = get_dataset('texas')
    data = dataset[0]

    perms = 3
    Ps = generate_permutation_matrices(size=data.x.size(0), amount=perms)

    args = get_test_config().copy()
    args['graph_size'] = data.size(0)
    args['input_dim'] = dataset.num_features
    args['output_dim'] = dataset.num_classes
    args['d'] = d
    args['normalised'] = True
    args['orth'] = orth
    args['edge_weights'] = True

    # Construct the model
    for i in range(perms):
        with torch.no_grad():
            model = DiscreteBundleSheafDiffusion(data.edge_index, args)
            model.eval()
            out = model(data.x)
            out = torch.FloatTensor(Ps[i] @ out.numpy())

            perm_data = permute_graph(data.clone(), Ps[i])
            model.update_edge_index(perm_data.edge_index)
            perm_out = model(perm_data.x)

        assert torch.allclose(out, perm_out, atol=1e-6)


@pytest.mark.parametrize("d, normalised", [(3, True), (5, True), (5, False),
                                           (7, True), (6, False), (4, True), (7, False)])
def test_permutation_equivariance_of_general_sheaf_diffusion_on_texas(d, normalised):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    dataset = get_dataset('texas')
    data = dataset[0]

    perms = 5
    Ps = generate_permutation_matrices(size=data.x.size(0), amount=perms)

    args = get_test_config().copy()
    args['graph_size'] = data.size(0)
    args['input_dim'] = dataset.num_features
    args['output_dim'] = dataset.num_classes
    args['d'] = d
    args['normalised'] = normalised
    args['deg_normalised'] = not normalised

    # Construct the model
    for i in range(perms):
        with torch.no_grad():
            model = DiscreteGeneralSheafDiffusion(data.edge_index, args)
            model.eval()
            assert not model.training
            assert not model.laplacian_builder.training
            out = model(data.x)
            out = torch.FloatTensor(Ps[i].astype(np.float64) @ out.numpy().astype(np.float64))

            perm_data = permute_graph(data.clone(), Ps[i])
            model.update_edge_index(perm_data.edge_index)
            perm_out = model(perm_data.x)

        assert torch.allclose(out, perm_out, atol=1e-4)


@pytest.mark.parametrize("d, orth", [(2, "euler"), (3, "euler"), (5, "matrix_exp"), (6, "cayley"), (7, "householder")])
def test_bundle_diffusion_backprop(d, orth):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)
    num_feat = 13

    # Build a test graph
    graph = nx.graph_atlas(546)
    data = from_networkx(graph)
    data.x = torch.FloatTensor(size=(len(graph), num_feat)).uniform_(-1.0, 1.0)
    data.y = torch.tensor([1]*len(graph), dtype=torch.long)

    # Construct the model
    args = get_test_config().copy()
    args['graph_size'] = data.size(0)
    args['input_dim'] = num_feat
    args['output_dim'] = 2
    args['d'] = d
    args['normalised'] = True
    args['orth'] = orth

    model = DiscreteBundleSheafDiffusion(data.edge_index, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)

    optimizer.zero_grad()
    out = model(data.x)
    assert list(out.size()) == [len(graph), 2]
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()

    # Check that gradients are back-propagated through the Laplacian in particular
    for param in model.parameters():
        assert param.grad is not None





