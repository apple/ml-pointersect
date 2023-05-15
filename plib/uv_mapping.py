#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class UVMap:
    def __init__(
            self,
            texture: np.ndarray,
            mode: str = 'wrap',
    ):
        """
        Args:
            texture:
                (h, w, dim)  for example, an rgb image, a displacement map, a bump map, etc
            mode:
                'wrap': used when 1 <= uv or uv <= 0.
                'edge': used when no wrapping is needed
        """
        self.texture = texture
        self.texture_height = self.texture.shape[0]
        self.texture_width = self.texture.shape[1]
        self.mode = mode

        # handle padding
        pad_widths = [[0, 0]] * self.texture.ndim
        pad_widths[0] = [1, 1]
        pad_widths[1] = [1, 1]
        padded_texture = np.pad(self.texture, pad_width=pad_widths, mode=mode)

        # create interpolator for the texture
        ys = np.linspace(-1, self.texture_height, self.texture_height + 2)  # 0, 1, ..., h-1
        xs = np.linspace(-1, self.texture_width, self.texture_width + 2)  # 0, 1, ..., w-1
        # yg, xg = np.meshgrid(ys, xs, indexing='ij')
        self.interpolator = RegularGridInterpolator(
            (ys, xs), padded_texture, method='linear', bounds_error=True)
        # image grid defined on 0..h-1

    def __call__(self, uv: np.ndarray):
        """
        query the texture map at locations uv
        Args:
            uv: (*, 2)  u is in the x/width direction, v is in the y/height direction,

        Returns:
            (*, dim)
        """
        if isinstance(uv, (list, tuple)):
            uv = np.array(uv)

        # in case want to tile the texture map
        uv = np.mod(uv, 1)

        # convert uv to yx
        y = uv[..., 1:2] * self.texture_height - 0.5  # (*, 1)
        x = uv[..., 0:1] * self.texture_width - 0.5  # (*, 1)
        yx = np.concatenate((y, x), axis=-1)
        return self.interpolator(yx)
