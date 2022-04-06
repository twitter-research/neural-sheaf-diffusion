# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import torch

from torch import nn
from torch_householder import torch_householder_orgqr


class Orthogonal(nn.Module):
    """Based on https://pytorch.org/docs/stable/_modules/torch/nn/utils/parametrizations.html#orthogonal"""
    def __init__(self, d, orthogonal_map):
        super().__init__()
        assert orthogonal_map in ["matrix_exp", "cayley", "householder", "euler"]
        self.d = d
        self.orthogonal_map = orthogonal_map

    def get_2d_rotation(self, params):
        # assert params.min() >= -1.0 and params.max() <= 1.0
        assert params.size(-1) == 1
        sin = torch.sin(params * 2 * math.pi)
        cos = torch.cos(params * 2 * math.pi)
        return torch.cat([cos, -sin,
                          sin, cos], dim=1).view(-1, 2, 2)

    def get_3d_rotation(self, params):
        assert params.min() >= -1.0 and params.max() <= 1.0
        assert params.size(-1) == 3

        alpha = params[:, 0].view(-1, 1) * 2 * math.pi
        beta = params[:, 1].view(-1, 1) * 2 * math.pi
        gamma = params[:, 2].view(-1, 1) * 2 * math.pi

        sin_a, cos_a = torch.sin(alpha), torch.cos(alpha)
        sin_b, cos_b = torch.sin(beta),  torch.cos(beta)
        sin_g, cos_g = torch.sin(gamma), torch.cos(gamma)

        return torch.cat(
            [cos_a*cos_b, cos_a*sin_b*sin_g - sin_a*cos_g, cos_a*sin_b*cos_g + sin_a*sin_g,
             sin_a*cos_b, sin_a*sin_b*sin_g + cos_a*cos_g, sin_a*sin_b*cos_g - cos_a*sin_g,
             -sin_b, cos_b*sin_g, cos_b*cos_g], dim=1).view(-1, 3, 3)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        if self.orthogonal_map != "euler":
            # Construct a lower diagonal matrix where to place the parameters.
            offset = -1 if self.orthogonal_map == 'householder' else 0
            tril_indices = torch.tril_indices(row=self.d, col=self.d, offset=offset, device=params.device)
            new_params = torch.zeros(
                (params.size(0), self.d, self.d), dtype=params.dtype, device=params.device)
            new_params[:, tril_indices[0], tril_indices[1]] = params
            params = new_params

        if self.orthogonal_map == "matrix_exp" or self.orthogonal_map == "cayley":
            # We just need n x k - k(k-1)/2 parameters
            params = params.tril()
            A = params - params.transpose(-2, -1)
            # A is skew-symmetric (or skew-hermitian)
            if self.orthogonal_map == "matrix_exp":
                Q = torch.matrix_exp(A)
            elif self.orthogonal_map == "cayley":
                # Computes the Cayley retraction (I+A/2)(I-A/2)^{-1}
                Id = torch.eye(self.d, dtype=A.dtype, device=A.device)
                Q = torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))
        elif self.orthogonal_map == 'householder':
            eye = torch.eye(self.d, device=params.device).unsqueeze(0).repeat(params.size(0), 1, 1)
            A = params.tril(diagonal=-1) + eye
            Q = torch_householder_orgqr(A)
        elif self.orthogonal_map == 'euler':
            assert 2 <= self.d <= 3
            if self.d == 2:
                Q = self.get_2d_rotation(params)
            else:
                Q = self.get_3d_rotation(params)
        else:
            raise ValueError(f"Unsupported transformations {self.orthogonal_map}")
        return Q