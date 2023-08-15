# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import torch_sparse

from torch import nn
from models.sheaf_base import SheafDiffusion
from models import laplacian_builders as lb
from models.sheaf_models import LocalConcatSheafLearner, EdgeWeightLearner
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint


class LaplacianODEFunc(nn.Module):
    """Implements Laplacian-based diffusion."""

    def __init__(
        self,
        d,  # final_d
        sheaf_learner,
        laplacian_builder,
        edge_index,
        graph_size,  # num nodes
        hidden_channels,  # number of feature channels (f)
        left_weights=False,
        right_weights=False,
        use_act=False,
        nonlinear=False,
        weight_learner=None,
    ):
        """
        Args:
            L: A sparse Laplacian matrix.
        """
        super(LaplacianODEFunc, self).__init__()
        self.d = d
        self.hidden_channels = hidden_channels
        self.weight_learner = weight_learner
        self.sheaf_learner = sheaf_learner
        self.laplacian_builder = laplacian_builder
        self.edge_index = edge_index
        self.nonlinear = nonlinear
        self.graph_size = graph_size
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.use_act = use_act
        # cache for the sheaf laplacian matrix
        self.L = None

        # W_1 (reweights stalk embeddings)
        if self.left_weights:
            self.lin_left_weights = nn.Linear(self.d, self.d, bias=False)
        # W_2 (controls channel numbers)
        if self.right_weights:
            self.lin_right_weights = nn.Linear(
                self.hidden_channels, self.hidden_channels, bias=False
            )

    def update_laplacian_builder(self, laplacian_builder):
        self.edge_index = laplacian_builder.edge_index
        self.laplacian_builder = laplacian_builder

    def forward(self, t, x):
        # Update the laplacian at each step.
        if self.nonlinear or self.L is None:
            # -> (n, d*f)
            x_maps = x.view(self.graph_size, -1)

            # build restriction maps
            maps = self.sheaf_learner(x_maps, self.edge_index)

            # compute edge weights
            if self.weight_learner is not None:
                edge_weights = self.weight_learner(x_maps, self.edge_index)
                L, _ = self.laplacian_builder(maps, edge_weights)
            else:
                L, _ = self.laplacian_builder(maps)
            self.L = L
        # Cache the Laplacian obtained at the first layer for the rest of the integration.
        else:
            L = self.L

        # apply left and right weights to features
        if self.left_weights:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_weights(x)
            x = x.reshape(-1, self.graph_size * self.d).t()

        if self.right_weights:
            x = self.lin_right_weights(x)

        x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), -x)

        if self.use_act:
            x = F.elu(x)

        return x


class ODEBlock(nn.Module):
    """Module performing the ODE Integration.
    (basically, a wrapper over `torchdiffeq.odeint`)
    """

    def __init__(self, odefunc, t, opt):
        super(ODEBlock, self).__init__()
        self.t = t  # == [0.0, self.t]
        self.opt = opt  # == args
        self.odefunc = odefunc
        self.set_tol()

    def set_tol(self):
        self.atol = self.opt["tol_scale"] * 1e-7
        self.rtol = self.opt["tol_scale"] * 1e-9
        if self.opt["adjoint"]:
            self.atol_adjoint = self.opt["tol_scale_adjoint"] * 1e-7
            self.rtol_adjoint = self.opt["tol_scale_adjoint"] * 1e-9

    def reset_tol(self):
        self.atol = 1e-7
        self.rtol = 1e-9
        self.atol_adjoint = 1e-7
        self.rtol_adjoint = 1e-9

    def forward(self, x):
        if self.opt["adjoint"] and self.training:
            z = odeint_adjoint(
                self.odefunc,
                x,
                self.t,
                method=self.opt["int_method"],
                options=dict(step_size=self.opt["step_size"], max_iters=self.opt["max_iters"]),
                adjoint_method=self.opt["adjoint_method"],
                adjoint_options=dict(
                    step_size=self.opt["adjoint_step_size"], max_iters=self.opt["max_iters"]
                ),
                atol=self.atol,
                rtol=self.rtol,
                adjoint_atol=self.atol_adjoint,
                adjoint_rtol=self.rtol_adjoint,
            )
        else:
            z = odeint(
                self.odefunc,
                x,
                self.t,
                method=self.opt["int_method"],
                options=dict(step_size=self.opt["step_size"], max_iters=self.opt["max_iters"]),
                atol=self.atol,
                rtol=self.rtol,
            )
        self.odefunc.L = None
        z = z[1]
        return z


class GraphLaplacianDiffusion(SheafDiffusion):
    """This is a diffusion model based on the weighted graph Laplacian."""

    def __init__(self, edge_index, args):
        super(GraphLaplacianDiffusion, self).__init__(edge_index, args)
        assert args["d"] == 1

        # linear layers
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        # sheaf learner = here, just learns edge weights from node features
        self.sheaf_learner = EdgeWeightLearner(self.hidden_dim, edge_index)

        # laplacian builder
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp, add_lp=self.add_lp
        )

        # odefunc defines our dy/dt
        self.odefunc = LaplacianODEFunc(
            self.final_d,
            self.sheaf_learner,
            self.laplacian_builder,
            edge_index,
            self.graph_size,
            self.hidden_channels,
            nonlinear=self.nonlinear,
            left_weights=self.left_weights,
            right_weights=self.right_weights,
            use_act=self.use_act,
        )
        # time_range: [0.0, max_t]
        self.odeblock = ODEBlock(self.odefunc, self.time_range, args)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)
        self.sheaf_learner.update_edge_index(edge_index)

    def forward(self, x):
        """x: node features"""

        x = F.dropout(x, p=self.input_dropout, training=self.training)

        # input dim -> hidden_dim
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # second hidden layer
        if self.second_linear:
            x = self.lin12(x)

        if self.t > 0:  # integrate ODE
            # -> (nd, f)
            x = x.view(self.graph_size * self.final_d, -1)
            # integrate -> get `x_i` from `x_0 = x`
            x = self.odeblock(x)

        # -> (n, d*f)
        x = x.view(self.graph_size, -1)

        # output layer
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)


class DiagSheafDiffusion(SheafDiffusion):
    """Performs diffusion using a sheaf Laplacian with diagonal restriction maps."""

    def __init__(self, edge_index, args):
        super(DiagSheafDiffusion, self).__init__(edge_index, args)

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim * self.graph_size, self.inter_dim)
        self.final_lin = nn.Linear(self.inter_dim, self.output_dim)

        self.sheaf_learner = LocalConcatSheafLearner(
            self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act
        )
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size,
            edge_index,
            d=self.d,
            normalised=self.normalised,
            deg_normalised=self.deg_normalised,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
        )

        self.odefunc = LaplacianODEFunc(
            self.final_d,
            self.sheaf_learner,
            self.laplacian_builder,
            edge_index,
            self.graph_size,
            self.hidden_channels,
            nonlinear=self.nonlinear,
            left_weights=self.left_weights,
            right_weights=self.right_weights,
            use_act=self.use_act,
        )
        self.odeblock = ODEBlock(self.odefunc, self.time_range, args)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)

    def forward(self, data):
        x = data.x

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        if self.t > 0:
            x = x.view(self.graph_size * self.final_d, -1)
            x = self.odeblock(x)
        x = x.view(self.graph_size, -1)

        # concat node embeddings
        x = x.reshape(x.size(0) // self.graph_size, -1)

        # 2-layer clf head
        x = self.lin2(F.elu(x))
        x = self.final_lin(F.elu(x))

        # return F.log_softmax(x, dim=1)
        return x


class BundleSheafDiffusion(SheafDiffusion):
    """Performs diffusion using a sheaf Laplacian with diagonal restriction maps."""

    def __init__(self, edge_index, args):
        super(BundleSheafDiffusion, self).__init__(edge_index, args)
        # Should use diagonal sheaf diffusion instead if d=1.
        assert args["d"] > 1

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim * self.graph_size, self.inter_dim)
        self.final_lin = nn.Linear(self.inter_dim, self.output_dim)

        self.weight_learner = (
            EdgeWeightLearner(self.hidden_dim, edge_index) if self.use_edge_weights else None
        )
        self.sheaf_learner = LocalConcatSheafLearner(
            self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act
        )
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size,
            edge_index,
            d=self.d,
            add_hp=self.add_hp,
            add_lp=self.add_lp,
            orth_map=self.orth_trans,
        )

        self.odefunc = LaplacianODEFunc(
            self.final_d,
            self.sheaf_learner,
            self.laplacian_builder,
            edge_index,
            self.graph_size,
            self.hidden_channels,
            nonlinear=self.nonlinear,
            left_weights=self.left_weights,
            right_weights=self.right_weights,
            use_act=self.use_act,
            weight_learner=self.weight_learner,
        )
        self.odeblock = ODEBlock(self.odefunc, self.time_range, args)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)
        self.weight_learner.update_edge_index(edge_index)

    def get_param_size(self):
        if self.orth_trans in ["matrix_exp", "cayley"]:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def forward(self, data):
        x = data.x

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        if self.t > 0:
            x = x.view(self.graph_size * self.final_d, -1)
            x = self.odeblock(x)

        x = x.view(self.graph_size, -1)

        # concat node embeddings
        x = x.reshape(x.size(0) // self.graph_size, -1)

        # 2-layer clf head
        x = self.lin2(F.elu(x))
        x = self.final_lin(F.elu(x))

        # return F.log_softmax(x, dim=1)
        return x


class GeneralSheafDiffusion(SheafDiffusion):
    def __init__(self, edge_index, args):
        super(GeneralSheafDiffusion, self).__init__(edge_index, args)
        # Should use diagoal diffusion if d == 1
        assert args["d"] > 1

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim * self.graph_size, self.inter_dim)
        self.final_lin = nn.Linear(self.inter_dim, self.output_dim)

        self.sheaf_learner = LocalConcatSheafLearner(
            self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act
        )
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size,
            edge_index,
            d=self.d,
            add_lp=self.add_lp,
            add_hp=self.add_hp,
            normalised=self.normalised,
            deg_normalised=self.deg_normalised,
        )

        self.odefunc = LaplacianODEFunc(
            self.final_d,
            self.sheaf_learner,
            self.laplacian_builder,
            edge_index,
            self.graph_size,
            self.hidden_channels,
            nonlinear=self.nonlinear,
            left_weights=self.left_weights,
            right_weights=self.right_weights,
            use_act=self.use_act,
        )
        self.odeblock = ODEBlock(self.odefunc, self.time_range, args)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)

    def forward(self, data):
        x = data.x

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        if self.t > 0:
            x = x.view(self.graph_size * self.final_d, -1)
            x = self.odeblock(x)

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.view(self.graph_size, -1)

        # concat node embeddings
        x = x.reshape(x.size(0) // self.graph_size, -1)

        # 2-layer clf head
        x = self.lin2(F.elu(x))
        x = self.final_lin(F.elu(x))

        #return F.log_softmax(x, dim=1)
        return x
