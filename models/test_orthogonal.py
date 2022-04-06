# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import numpy as np

from models.orthogonal import Orthogonal


@pytest.mark.parametrize("d, orth_trans", [(2, "euler"), (3, "euler"), (4, "householder"), (2, "householder"),
                                           (5, "matrix_exp"), (6, "cayley")])
def test_orthogonal_transformations(d, orth_trans):
    # Fix the random seed
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Construct the map parameters
    samples = 200
    if orth_trans in ["householder", "euler"]:
        x = torch.FloatTensor(size=(samples, d * (d - 1) // 2)).uniform_(-1.0, 1.0)
    else:
        x = torch.FloatTensor(size=(samples, d * (d + 1) // 2)).uniform_(-1.0, 1.0)

    # Construct the model
    orth_transform = Orthogonal(d=d, orthogonal_map=orth_trans)
    maps = orth_transform(x)

    assert list(maps.size()) == [samples, d, d]

    # Check that the matrices are orthogonal.
    A = torch.bmm(maps.transpose(-2, -1), maps)
    Id = torch.eye(d).unsqueeze(0).expand(samples, -1, -1)
    assert torch.allclose(A, Id, atol=1e-6)

    # We only obtain SO(n) matrices here.
    det = torch.linalg.det(A)
    assert torch.allclose(det, torch.ones_like(det))




