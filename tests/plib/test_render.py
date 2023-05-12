import unittest
from plib import utils
from plib.render import generate_point
import torch
import numpy as np
from scipy.spatial.transform import Rotation


class Test_compute_3d_xyz(unittest.TestCase):

    def _get_random_setup(self):
        intrinsics = np.array([
            [128., 0, 128.],
            [0., 128., 128.],
            [0., 0., 1.],
        ])
        w = 256
        h = 256
        z_map = np.random.rand(h, w) * 10. + 1.
        img = np.random.rand(h, w, 3)
        r_angles = np.random.rand(3) * 60.  # in degree
        R = Rotation.from_euler('xyz', r_angles, degrees=True).as_matrix()  # (3,3) rotation matrix
        t = np.random.rand(3)
        H_c2w = np.eye(4)
        H_c2w[:3, :3] = R
        H_c2w[:3, 3] = t

        return dict(
            intrinsics=intrinsics,
            z_map=z_map,
            img=img,
            H_c2w=H_c2w,
        )

    def _test(self, subsample: int):
        info_dict = self._get_random_setup()
        points1, _ = generate_point(
            rgb_image=info_dict['img'],
            depth_image=info_dict['z_map'],
            intrinsic=info_dict['intrinsics'],
            subsample=subsample,
            world_coordinate=True,
            pose=info_dict['H_c2w'],
        )  # (N, 3)

        info_dict_t = utils.to_tensor(info_dict)
        points2 = utils.compute_3d_xyz(
            z_map=info_dict_t['z_map'],
            intrinsic=info_dict_t['intrinsics'],
            H_c2w=info_dict_t['H_c2w'],
            subsample=subsample,
        )  # (h, w, 3)
        points2 = points2.reshape(-1, 3)

        assert torch.allclose(torch.from_numpy(points1), points2)

    def test_1(self):
        self._test(subsample=1)

    def test_2(self):
        self._test(subsample=2)




if __name__ == '__main__':
    unittest.main()
