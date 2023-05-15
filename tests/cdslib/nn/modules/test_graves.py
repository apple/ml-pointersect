#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#

import unittest
import torch
from cdslib.core.nn import NetworkGraves
from cdslib.core.nn import detach
import typing as T

class MyTest_Graves(unittest.TestCase):

    def test_1(self):
        self._build_model()

    def test_2(self):
        self._build_model(
            dim_z=10,
            output_layer_type='StackedModulatedLinearLayers',
            feed_z_to_decode_rnn=True,
            num_output_layers=3,
            dim_output_layers=128,
            output_layer_nonlinearity='relu',
            output_layer_dropout_prob=0,
            output_layer_normalize=True,
            output_layer_normalize_first=True,
            output_layer_init_freq_first_layer=1.,
            output_layer_init_freq_other_layer=25.,
            # style mapping layer
            num_mapping_layers=4,
            dim_mapping_layers=256,
            mapping_layer_nonlinearity='silu',
            mapping_layer_dropout_prob=0.,
        )


    def test_3(self):
        self._build_model(
            dim_z=20,
            output_layer_type='StackedFiLMLayers',
            feed_z_to_decode_rnn=True,
            num_output_layers=3,
            dim_output_layers=128,
            output_layer_nonlinearity='relu',
            output_layer_dropout_prob=0,
            output_layer_normalize=True,
            output_layer_normalize_first=True,
            output_layer_init_freq_first_layer=1.,
            output_layer_init_freq_other_layer=25.,
            # style mapping layer
            num_mapping_layers=4,
            dim_mapping_layers=256,
            mapping_layer_nonlinearity='silu',
            mapping_layer_dropout_prob=0.,
        )


    def _build_model(
            self,
            batch_size=10,
            seq_len = 20,
            num_chars = 7,
            dim_x = 5,
            dim_c = 6,
            dim_z = 0,
            dim_p = 9,
            output_layer_type: str = 'StackedLinearLayers',
            feed_z_to_decode_rnn: bool = True,
            num_output_layers: int = 1,  # int
            dim_output_layers: T.Union[int, T.Sequence[int]] = 128,
            # T.Union[int, T.Sequence[int]], len = num_output_layers-1
            output_layer_nonlinearity='relu',
            output_layer_add_norm_layer=False,
            output_layer_norm_fun: T.Callable = torch.nn.LayerNorm,
            output_layer_dropout_prob: float = 0,
            output_layer_normalize: bool = True,
            output_layer_normalize_first: bool = True,
            output_layer_init_freq_first_layer: float = 1.,
            output_layer_init_freq_other_layer: float = 25.,
            # style mapping layer
            num_mapping_layers: int = 4,
            dim_mapping_layers: int = 256,
            mapping_layer_nonlinearity: str = 'silu',
            mapping_layer_dropout_prob: float = 0.,
    ):

        net = NetworkGraves(
            dim_x=dim_x,
            dim_c=dim_c,
            dim_p=dim_p,
            dim_z=dim_z,
            output_layer_type=output_layer_type,
            feed_z_to_decode_rnn=feed_z_to_decode_rnn,
            num_output_layers=num_output_layers,
            dim_output_layers=dim_output_layers,
            output_layer_nonlinearity=output_layer_nonlinearity,
            output_layer_add_norm_layer=output_layer_add_norm_layer,
            output_layer_dropout_prob=output_layer_dropout_prob,
            output_layer_normalize=output_layer_normalize,
            output_layer_normalize_first=output_layer_normalize_first,
            output_layer_init_freq_first_layer=output_layer_init_freq_first_layer,
            output_layer_init_freq_other_layer=output_layer_init_freq_other_layer,
            # style mapping layer
            num_mapping_layers=num_mapping_layers,
            dim_mapping_layers=dim_mapping_layers,
            mapping_layer_nonlinearity=mapping_layer_nonlinearity,
            mapping_layer_dropout_prob=mapping_layer_dropout_prob,
        )

        xs = torch.randn(seq_len, batch_size, dim_x)
        cs = torch.randn(num_chars, batch_size, dim_c)
        init_h = net.get_zero_hidden_states(batch_size=batch_size, device=xs.device)


        if dim_z == 0:
            zt_fun = None
        else:
            def zt_fun(avail_inputs):
                return dict(
                    zs=torch.randn(seq_len, batch_size, dim_z),
                )

        # forward
        out_dict = net(xs=xs, cs=cs, init_hidden_states=init_h, zt_fun=zt_fun)

        # check ps
        ps = out_dict['ps']
        assert len(ps.shape) == 3
        assert ps.size(0) == seq_len
        assert ps.size(1) == batch_size
        assert ps.size(2) == dim_p

        # check attn_cs
        attn_cs = out_dict['attn_cs']
        assert len(attn_cs.shape) == 3
        assert attn_cs.size(0) == seq_len
        assert attn_cs.size(1) == batch_size
        assert attn_cs.size(2) == dim_c

        # check attn_weights
        attn_weights = out_dict['attn_weights']
        assert len(attn_weights.shape) == 3
        assert attn_weights.size(0) == seq_len
        assert attn_weights.size(1) == batch_size
        assert attn_weights.size(2) == num_chars + 1

        # check attn_hs
        attn_hs = out_dict['attn_hs']
        assert len(attn_hs.shape) == 3
        assert attn_hs.size(0) == seq_len
        assert attn_hs.size(1) == batch_size
        assert attn_hs.size(2) == net.dim_attn_rnn_h

        # check decode_hs
        decode_hs = out_dict['decode_hs']
        assert len(decode_hs.shape) == 3
        assert decode_hs.size(0) == seq_len
        assert decode_hs.size(1) == batch_size
        assert decode_hs.size(2) == net.dim_decode_rnn_h

        # check hidden state
        hidden_states = out_dict['hidden_states']
        for key in ['attn_rnn_h',
                    'decode_rnn_h',
                    'attn_c',
                    'attn_mean_idxs']:
            assert key in hidden_states

        # simple loss and backward
        loss = ps.sum()
        loss.backward()

        # use the final_hidden_states to initialize a new forward
        # to make sure they are of the right shape and format
        init_h = detach(out_dict['hidden_states'])
        out_dict = net(xs=xs, cs=cs, init_hidden_states=init_h, zt_fun=zt_fun)

        ps = out_dict['ps']
        assert len(ps.shape) == 3
        assert ps.size(0) == seq_len
        assert ps.size(1) == batch_size
        assert ps.size(2) == dim_p

        # simple loss and backward
        loss = ps.sum()
        loss.backward()


if __name__ == '__main__':
    unittest.main()
