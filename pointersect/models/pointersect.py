#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
import math
import typing as T
import warnings

import numpy as np
import torch

from cdslib.core.nn import StackedLinearLayers
from plib import utils
from pointersect.models import network_transformer, model_utils


class SimplePointersect(torch.nn.Module):
    """
    This is just a simple transformer with a learned token whose
    output token is used to infer t and surface normal.
    """

    def __init__(
            self,
            learn_dist: bool,
            num_layers: int,
            dim_feature: int,
            num_heads: int,
            positional_encoding_num_functions: int,
            positional_encoding_include_input: bool,
            positional_encoding_log_sampling: bool,
            nonlinearity: str,
            dim_mlp: int,
            encoding_type: str = 'pos',  # currently only support classic positional encoding
            dropout: float = 0.1,
            direction_param: str = 'norm_vec',  # 'theta_phi'
            estimate_surface_normal_weights: bool = True,
            estimate_image_rendering_weights: bool = False,
            use_layer_norm: bool = False,
            dim_point_feature: int = 0,  # additional feature description of points (other than xyz)
            use_rgb_as_input: bool = False,
            use_dist_as_input: bool = False,  # if true, use |x|,|y|,|z| and sqrt(x^2+y^2) in ray space as input
            use_zdir_as_input: bool = False,  # if true, use camera viewing direction (2 vector, 3 dim) as input
            use_dps_as_input: bool = False,  # if true, use local frame width (1 value, 1 dim) as input
            use_dpsuv_as_input: bool = False,  # if true, use local frame (2 vectors, 6 dim) as input
            use_pr: bool = False,  # if true, learn a token to replace invalid input
            use_additional_invalid_token: bool = False,  # if true, an invalid_token is added as a k+1 th input
            dim_input_layers: T.List[int] = None,  # dimension of the linear layers (nLayer-1)
            use_vdir_as_input: bool = False,  # if true, use camera viewing direction (1 vector, 3 dim) as input
            use_rgb_indicator: bool = False,  # whether to add a binary indicator saying input has valid rgb
            use_feature_indicator: bool = False,  # whether to add a binary indicator saying input has valid feature
            weight_layer_type: str = 'multihead',
            # 'multihead' or 'dot_prod', the layer type at the end to compute the combination weights
    ):
        super().__init__()

        self.learn_dist = learn_dist
        self.num_layers = num_layers
        self.dim_feature = dim_feature
        self.num_heads = num_heads
        self.encoding_type = encoding_type
        self.positional_encoding_num_functions = positional_encoding_num_functions
        self.positional_encoding_include_input = positional_encoding_include_input
        self.positional_encoding_log_sampling = positional_encoding_log_sampling
        self.nonlinearity = nonlinearity
        self.dim_mlp = dim_mlp
        self.direction_param = direction_param
        self.estimate_surface_normal_weights = estimate_surface_normal_weights
        self.estimate_image_rendering_weights = estimate_image_rendering_weights
        self.use_layer_norm = use_layer_norm
        self.dim_point_feature = dim_point_feature
        self.use_rgb_as_input = use_rgb_as_input
        self.use_dist_as_input = use_dist_as_input
        self.use_zdir_as_input = use_zdir_as_input
        self.use_dps_as_input = use_dps_as_input
        self.use_dpsuv_as_input = use_dpsuv_as_input
        self.use_pr = use_pr
        self.use_additional_invalid_token = use_additional_invalid_token
        self.dim_input_layers = dim_input_layers
        self.use_vdir_as_input = use_vdir_as_input
        self.use_rgb_indicator = use_rgb_indicator
        self.use_feature_indicator = use_feature_indicator
        self.weight_layer_type = weight_layer_type

        if self.direction_param == 'norm_vec':
            self.dim_direction = 3
        elif self.direction_param == 'theta_phi':
            self.dim_direction = 2
        else:
            raise NotImplementedError

        if self.learn_dist:
            self.dim_out = 2 + (self.dim_direction + 1) + 1  # t: (mean, std), normal: (direction, k), hit_logit
        else:
            self.dim_out = 1 + self.dim_direction + 1  # (t + surface normal as (theta, phi) + hit_logit)
        self.dropout = dropout

        # construct positional embedding function
        if encoding_type == 'pos':
            # Note: although we call the function, we set the num_encoding_functions to `0`
            # and use the input as is.
            self.pos_embedder = model_utils.get_embedding_function(
                num_encoding_functions=self.positional_encoding_num_functions,
                include_input=self.positional_encoding_include_input,
                log_sampling=self.positional_encoding_log_sampling,
            )
            self.dim_pos = 3 * (
                        int(self.positional_encoding_include_input) + self.positional_encoding_num_functions * 2)
        else:
            raise NotImplementedError

        self.dim_input = self.dim_pos + self.dim_point_feature

        if self.nonlinearity == 'leaky_relu':
            self.nonlinearity_fun = lambda: torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        elif self.nonlinearity == 'relu':
            self.nonlinearity_fun = lambda: torch.nn.ReLU(inplace=False)
        elif self.nonlinearity == 'tanh':
            self.nonlinearity_fun = torch.nn.Tanh
        elif self.nonlinearity == 'sigmoid':
            self.nonlinearity_fun = torch.nn.Sigmoid
        elif self.nonlinearity in {'silu', 'swish'}:
            self.nonlinearity_fun = lambda: torch.nn.SiLU(inplace=False)
        else:
            raise ValueError('unsupported nonlinearity')

        if self.use_additional_invalid_token and not self.use_pr:
            raise ValueError('learned_invalid_input_token is bound with pr')

        # a linear layer to transform positional encoding to feature
        if self.dim_input_layers is None:
            self.input_linear = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self.dim_input,
                    out_features=self.dim_feature,
                    bias=True
                ),
                self.nonlinearity_fun(),
            )
        else:
            if isinstance(self.dim_input_layers, int):
                self.dim_input_layers = [self.dim_input_layers]
            self.input_linear = StackedLinearLayers(
                num_layers=len(self.dim_input_layers) + 1,
                dim_input=self.dim_input,
                dim_output=self.dim_feature,
                dim_features=self.dim_input_layers,
                nonlinearity=self.nonlinearity,
                add_norm_layer=False,
                output_add_nonlinearity=True,
            )

        # transformer
        layer = network_transformer.TransformerEncoderLayer(
            d_model=self.dim_feature,
            nhead=self.num_heads,
            dim_feedforward=self.dim_mlp,
            dropout=self.dropout,
            activation=self.nonlinearity_fun(),
            batch_first=True,
            use_layer_norm=self.use_layer_norm,
        )
        # self.transformer = torch.nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.transformer = network_transformer.TransformerEncoder(layer, num_layers=self.num_layers)

        # a learned token
        self.register_parameter(
            name='learned_token',
            param=torch.nn.Parameter(torch.randn(self.dim_feature)),
        )

        # another learned token in pr to replace invalid input
        if self.use_pr:
            self.register_parameter(
                name='learned_invalid_input_token',
                param=torch.nn.Parameter(torch.zeros(self.dim_feature)),
            )
        else:
            self.learned_invalid_input_token = None

        # a linear layer to transform output learned token to t and surface normal
        self.output_linear = torch.nn.Sequential(
            self.nonlinearity_fun(),
            torch.nn.Linear(
                in_features=self.dim_feature,
                out_features=self.dim_out,
                bias=True,
            ),
        )

        # a linear layer to compute the feature to compute the weight for surface normal
        if self.estimate_surface_normal_weights:
            if self.weight_layer_type == 'multihead':
                self.net_geometry = torch.nn.MultiheadAttention(
                    embed_dim=self.dim_feature,
                    num_heads=self.num_heads,
                    dropout=0.,
                    bias=True,
                    batch_first=True,
                )
            elif self.weight_layer_type == 'dot_prod':
                self.net_geometry = None
                warnings.warn(f'just to test speed, not implemented actually')
            else:
                raise NotImplementedError

        # a few layer to compute the weights for image-based rendering
        if self.estimate_image_rendering_weights:
            if self.weight_layer_type == 'multihead':
                self.net_blending = torch.nn.MultiheadAttention(
                    embed_dim=self.dim_feature,
                    num_heads=self.num_heads,
                    dropout=0.,
                    bias=True,
                    batch_first=True,
                )
            elif self.weight_layer_type == 'dot_prod':
                self.net_blending = None
                warnings.warn(f'just to test speed, not implemented actually')
            else:
                raise NotImplementedError

    def forward(
            self,
            points: torch.Tensor,
            additional_features: torch.Tensor = None,
            neighbor_num: torch.Tensor = None,
            valid_mask: torch.Tensor = None,
            printout: bool = False,
            max_chunk_size: int = -1,
            check_finite: bool = False,
    ):
        """
        Args:
            points: (b, m, k, 3) or (bm, k, 3),  m: number of rays, n: number of neighbor points, xyz coord of points
            additional_features:  (b, m, n, dim_point_feature)
            neighbor_num:
                (b, m)  number of valid neighbors found by pr_cuda.  dtype: long
                pr_cuda returns index (b, m, k), among the k points, only neighbor_num[b, m] is valid,
                others are padding with dummy index 0 (mapped to (10^12, 10^12, 10^12))
            valid_mask:
                (b, m, k) or (bm, k) whether the point is valid to use

        Returns:
            ts: (b, m)
            surface_normals: (b, m, 3)
        """

        if max_chunk_size < 0:
            max_chunk_size = np.inf

        *b_shape, n, dim = points.shape
        device = points.device

        # Note that when we only find neighbors within a fixed distance of ray and the number < n,
        # we use a far away dummy point (1e12,1e12,1e12) to the fill the neighbor position
        # so pr points may contain a far away one

        points = points.reshape(-1, points.size(-2), points.size(-1))  # (b*m, n, 3)

        if valid_mask is not None:
            valid_mask = valid_mask.reshape(-1, points.size(-2))  # (bm, n)

        # pr use neighbor_num to know which points are invalid
        if self.use_pr:
            assert neighbor_num is not None, "need to include neighbor num when use pr"

        assert torch.isfinite(points).all()
        # The position of points can actually be infinite. Since the ray may shoot to opposite direction to all points,
        # the t_s will all be negative and thus become inf.

        # get positional encoding
        xs = self.pos_embedder(points)  # (b*m, n, dim)

        if check_finite and (not torch.isfinite(xs).all() or torch.isnan(xs).any()):
            print('positional embedding is wrong!')

        # concat additional feature
        if additional_features is not None:
            xs = torch.cat(
                (
                    xs,  # (b*m, n, dim)
                    additional_features.reshape(-1, additional_features.size(-2), additional_features.size(-1)),
                # (b*m, n, d)
                ), dim=-1)  # (b*m, n, dim)

        if printout and torch.cuda.is_available():
            print(f'after pos encoding: {torch.cuda.memory_allocated(device=device) / 2 ** 30} GB')
            print(f"  xs: {xs.shape} {np.prod(xs.shape) * 4 / 2 ** 30:.2f} GB")

        # input linear
        xs = self.input_linear(xs)  # (b*m, n, dim)

        if check_finite and (not torch.isfinite(xs).all() or torch.isnan(xs).any()):
            print('input linear is wrong!')

        # use_pr is the param to determine whether pointersect will learn a token to replace invalid point
        # note that a far away dummy point (1e12,1e12,1e12) will still pass pos_embedder
        # but it will be detected by neighbor_num and replaced by learned token
        # also note that, while this dummy point and does give input position to pointersect
        # its position may still be used to determine translation
        # that case is handled by utils.rectify_points

        if self.use_pr:
            xs = xs.reshape(-1, self.dim_feature)  # (b*m*n, dim)
            # invalid_point = torch.logical_not(torch.isfinite(points).all(dim=-1)).reshape(-1)
            # invalid_point = (torch.abs(points) > 1e6).any(dim=-1).reshape(-1)

            neighbor_idx = torch.arange(n, device=neighbor_num.device).view(*([1] * len(b_shape)), -1).expand(
                *b_shape, -1)  # (b,m,n)
            invalid_point = (neighbor_idx >= neighbor_num.unsqueeze(-1)).reshape(-1)  # (b*m*n,)

            xs[invalid_point] = self.learned_invalid_input_token
            xs = xs.reshape(-1, n, self.dim_feature)  # (b*m, n, dim)

        if valid_mask is not None and self.learned_invalid_input_token is not None:
            xs[torch.logical_not(valid_mask)] = self.learned_invalid_input_token

        if check_finite and (not torch.isfinite(xs).all() or torch.isnan(xs).any()):
            print('invalid learned token is wrong!')

        # learned token is the first token
        xs = torch.cat(
            (
                self.learned_token.reshape(1, 1, self.dim_feature).expand(xs.size(0), -1, -1),
                # (dim,) -> (b*m, 1, dim)
                xs  # (b*m, 1, dim)
            ),
            dim=-2
        )

        # use the additional invalid token as k+1th input
        if self.use_additional_invalid_token:
            xs = torch.cat(
                (
                    xs,  # (b*m, 1, dim)
                    self.learned_invalid_input_token.reshape(1, 1, self.dim_feature).expand(xs.size(0), -1, -1),
                ),
                dim=-2
            )

        # print(f'learned token norm is {self.learned_token.norm()}')

        if check_finite and (not torch.isfinite(xs).all() or torch.isnan(xs).any()):
            print('learned token is wrong!')

        if printout and torch.cuda.is_available():
            print(f'after input linear: {torch.cuda.memory_allocated(device=device) / 2 ** 30} GB')
            print(f"  xs: {xs.shape} {np.prod(xs.shape) * 4 / 2 ** 30:.2f} GB")

        # transformer
        bm = xs.size(0)
        if bm > max_chunk_size:
            out_tokens = []
            out_weights = []
            num_chunks = math.ceil(bm / max_chunk_size)
            chunk_dim = 0
            xs_list = torch.chunk(xs, chunks=num_chunks, dim=chunk_dim)
            for xs in xs_list:
                # output attention weights as well
                chunked_tokens, chunked_weights = self.transformer(xs)
                out_tokens.append(chunked_tokens)  # (b*m, n, dim)
                out_weights.append(chunked_weights)  # (b*m, n+1, num_layers)

            out_tokens = torch.cat(out_tokens, dim=chunk_dim)  # (b*m, n, dim)
            out_weights = torch.cat(out_weights, dim=chunk_dim)

            del xs
        else:
            # out_tokens = self.transformer(xs)  # (b*m, n, dim)
            out_tokens, out_weights = self.transformer(xs)  # (b*m, n+1, num_layers)
            del xs

        if check_finite and (not torch.isfinite(out_tokens).all() or torch.isnan(out_tokens).any()):
            print('out_tokens are wrong!')

        if printout and torch.cuda.is_available():
            print(f'after transformer: {torch.cuda.memory_allocated(device=device) / 2 ** 30} GB')
            print(f"  out_tokens: {out_tokens.shape} {np.prod(out_tokens.shape) * 4 / 2 ** 30:.2f} GB")

        # remove the additional token
        if self.use_additional_invalid_token:
            out_tokens = out_tokens[..., :-1, :]

        learned_token_output = out_tokens[..., 0, :]  # (b*m, dim)

        out = self.output_linear(learned_token_output)  # (b*m, dim_out)

        if check_finite and (not torch.isfinite(out).all() or torch.isnan(out).any()):
            print('out are wrong!')

        out = out.reshape(*b_shape, self.dim_out)  # (b, m, dim_out)

        idx = 0
        est_t = out[..., idx]  # (b, m)
        idx += 1
        raw_direction = out[...,
                        idx:idx + self.dim_direction]  # (b, m, 2/3), angle on xy plane from x axis, angle againt xy plane
        idx += self.dim_direction
        est_hit_logit = out[..., idx]  # (b, m)
        idx += 1

        if self.learn_dist:
            est_t_log_std = out[..., idx]  # (b, m)
            idx += 1
            log_ks = out[..., idx]  # (b, m)
            idx += 1
        else:
            est_t_log_std = None
            log_ks = None

        if self.direction_param == 'theta_phi':
            assert raw_direction.size(-1) == 2
            r = torch.cos(raw_direction[..., 1])
            est_normal = torch.stack(
                (
                    r * torch.cos(raw_direction[..., 0]),
                    r * torch.sin(raw_direction[..., 0]),
                    torch.sin(raw_direction[..., 1]),
                ), dim=-1)  # (b, m, 3)
        elif self.direction_param == 'norm_vec':
            assert raw_direction.size(-1) == 3
            est_normal = torch.nn.functional.normalize(raw_direction, dim=-1, eps=1.e-8)
        else:
            raise NotImplementedError

        # make sure surface normal points to (0,0,0) (opposite direction of (0,0,1))
        est_normal_sign = torch.sign(est_normal[..., 2:3]).detach_()
        est_normal = est_normal * (-1 * est_normal_sign)

        # compute geometry weights
        if self.estimate_surface_normal_weights:
            if self.use_pr:
                warnings.warn(
                    "WARNING! With pr, invalid points are set to be points far away. "
                    "Consider disable plane_normals."
                )

            if valid_mask is not None:  # valid_mask: (bm, n)
                # if a ray has no valid points, the softmax in multihead attention will return
                # nan in output and grad if a bool attn_mask is used
                # => we just add a large negative number and handle the output weights
                attn_mask = torch.zeros(bm, n, dtype=out_tokens.dtype, device=out_tokens.device)  # (bm, n)
                attn_mask = attn_mask.masked_fill(~valid_mask, -1e9)
                attn_mask = attn_mask.view(bm, 1, 1, n).expand(
                    bm, self.num_heads, 1, n).reshape(-1, 1, n)  # (bm * num_head, 1, n)

                # if attn_mask is all true at the last dim,
                # the cooresponding attn_output and geometry_weights will be nan

                # tmp_out_tokens = out_tokens.clone()
                # tmp_out_tokens[..., 1:, :] = tmp_out_tokens[..., 1:, :].masked_fill(~valid_mask.unsqueeze(-1), 0)

            else:
                attn_mask = None
                # tmp_out_tokens = out_tokens

            if self.weight_layer_type == 'multihead':
                attn_output, geometry_weights = self.net_geometry(
                    query=out_tokens[..., 0:1, :],  # (b*m, 1, dim)
                    key=out_tokens[..., 1:, :],  # (b*m, n, dim)
                    value=out_tokens[..., 1:, :],  # (b*m, n, dim)
                    need_weights=True,
                    attn_mask=attn_mask,
                    # (bm*num_head, 1, n)  be careful using bool attn_mask causes nan, use float!!!!
                )  # attn_output: (b*m, 1, dim)   attn_output_weights: (b*m, 1, n)

                # add 0 * attn_output so that we do not need to detect unused parameters for DDP
                geometry_weights = geometry_weights + 0. * attn_output.mean()
            elif self.weight_layer_type == 'dot_prod':
                warnings.warn('dot_prod just to test speed')
                geometry_weights = torch.rand(bm, 1, n, device=out_tokens.device)
                geometry_weights = geometry_weights / geometry_weights.sum(dim=-1, keepdim=True)
                geometry_weights = geometry_weights.reshape(*b_shape, n)  # (b, m, n) positive, sum to 1

            else:
                raise NotImplementedError

            if valid_mask is not None:
                valid_est_plane_normal_mask = valid_mask.any(dim=-1)  # (bm, 1)
            else:
                valid_est_plane_normal_mask = torch.ones(bm, 1, dtype=torch.bool, device=points.device)

            centers = torch.zeros(*points.shape[:-2], 3, dtype=points.dtype, device=points.device)  # (b*m, 3)
            # (all rays have direction (0,0,1), centered at (0,0,0)), so the intersection point
            # is at (0,0,t)
            centers[..., 2:3] = est_t.reshape(-1, 1)
            plane_dict = utils.fit_hyperplane(
                points=points,  # (b*m, n, 3)
                centers=centers,  # (b*m, 3)
                weights=geometry_weights.squeeze(-2),  # (b*m, n)
            )  # (b*m, 3)
            est_plane_normal = plane_dict['plane_normals'].reshape(*b_shape, 3)  # (b, m, 3)
            valid_est_plane_normal_mask = torch.logical_and(
                valid_est_plane_normal_mask.reshape(*b_shape),  # (b, m)
                plane_dict['valid_mask'].reshape(*b_shape),  # (b, m)
            )  # (b, m)

            # make sure surface normal points to (0,0,0) (opposite direction of (0,0,1))
            est_normal_sign = torch.sign(est_plane_normal[..., 2:3]).detach_()
            est_plane_normal = est_plane_normal * (-1 * est_normal_sign)
            # move reshape in the returned dict to here. reshape should happen only with valid geometry_weights
            geometry_weights = geometry_weights.reshape(*b_shape, n)  # (b, m, n)
        else:
            est_plane_normal = None
            geometry_weights = None
            valid_est_plane_normal_mask = None

        if self.estimate_image_rendering_weights:
            if self.weight_layer_type == 'multihead':
                attn_output, blending_weights = self.net_blending(
                    query=out_tokens[..., 0:1, :],  # (b*m, 1, dim)
                    key=out_tokens[..., 1:, :],  # (b*m, n, dim)
                    value=out_tokens[..., 1:, :],  # (b*m, n, dim)
                    need_weights=True,
                )  # attn_output: (b*m, 1, dim)   blending_weights: (b*m, 1, n)
                blending_weights = blending_weights.reshape(*b_shape, n)  # (b, m, n) positive, sum to 1

                # add 0 * attn_output so that we do not need to detect unused parameters for DDP
                blending_weights = blending_weights + 0. * attn_output.mean()

            elif self.weight_layer_type == 'dot_prod':
                warnings.warn('dot_prod just to test speed')
                blending_weights = torch.rand(bm, 1, n, device=out_tokens.device)
                blending_weights = blending_weights / blending_weights.sum(dim=-1, keepdim=True)
                blending_weights = blending_weights.reshape(*b_shape, n)  # (b, m, n) positive, sum to 1

            else:
                raise NotImplementedError
        else:
            blending_weights = None

        return dict(
            ts=est_t,  # (b, m)
            surface_normals=est_normal,  # (b, m, 3)  # last dimension is positive, normalized
            hit_logits=est_hit_logit,  # (b, m)
            ts_log_std=est_t_log_std,  # (b, m)
            log_ks=log_ks,  # (b, m)
            est_plane_normals=est_plane_normal,  # (b, m, 3)
            geometry_weights=geometry_weights,  # (b, m, n)
            blending_weights=blending_weights,  # (b, m, n)
            out_weights=out_weights,  # (b, m, num_layer)
            valid_est_plane_normal_mask=valid_est_plane_normal_mask,  # (b, m)
        )
