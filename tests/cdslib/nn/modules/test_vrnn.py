#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#

import unittest
import torch
from cdslib.core.nn import ParamGraves, NetworkVRNN, ParamVRNN
from cdslib.core.nn import detach
import random


class MyTest_Graves(unittest.TestCase):
    def test_1(self):
        batch_size = 10
        seq_len = 20
        seq_len_fs = 15
        num_chars = 7
        dim_x = 5
        dim_c = 6
        dim_p = 9
        dim_z = 7
        dim_latent = 10
        dim_f = 13
        valid_lens_fs = [random.randint(1, seq_len_fs) for _ in range(batch_size)]

        param_graves = ParamGraves(dim_x=dim_x, dim_c=dim_c, dim_p=dim_p, dim_z=dim_z)
        param_vrnn = ParamVRNN(param_graves=param_graves, dim_latent=dim_latent, dim_f=dim_f)

        net = NetworkVRNN(param_vrnn)

        xs = torch.randn(seq_len, batch_size, dim_x)
        cs = torch.randn(num_chars, batch_size, dim_c)
        fs = torch.randn(seq_len_fs, batch_size, dim_f)
        init_h = net.get_zero_hidden_states(batch_size=batch_size, device=xs.device)

        # forward
        out_dict = net(xs=xs, cs=cs, fs=fs, init_hidden_states=init_h, valid_lens_fs=valid_lens_fs)

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
        out_dict = net(xs=xs, cs=cs, fs=fs, init_hidden_states=init_h, valid_lens_fs=valid_lens_fs)

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
