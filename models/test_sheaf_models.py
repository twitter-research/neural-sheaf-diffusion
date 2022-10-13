# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import numpy as np
import networkx as nx

from models.sheaf_models import LocalConcatSheafLearner
from torch_geometric.utils import from_networkx


@pytest.mark.parametrize("graph_id, in_channels, d", [(5, 10, 1), (70, 7, 4)])
def test_local_concat_learner(graph_id, in_channels, d):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Build a test graph
    graph = nx.graph_atlas(graph_id)
    data = from_networkx(graph)
    data.x = torch.FloatTensor(size=(len(graph), in_channels)).uniform_(-1.0, 1.0)

    # Construct the model
    model = LocalConcatSheafLearner(in_channels=in_channels, out_shape=(d, d))
    maps = model(data.x, data.edge_index)

    assert list(maps.size()) == [data.edge_index.size(1), d, d]

    # Construct the model
    model = LocalConcatSheafLearner(in_channels=in_channels, out_shape=(d**2,))
    maps = model(data.x, data.edge_index)

    assert list(maps.size()) == [data.edge_index.size(1), d**2]






