#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
import copy
import warnings

import numpy as np
import typing as T
from scipy.spatial.transform import Rotation
import torch
from plib import sample_utils


class RigidMotion:
    def __init__(self, R: np.ndarray = None, t: np.ndarray = None, H: np.ndarray = None):
        if H is not None:
            R = H[:3, :3]
            t = H[:3, 3]
        if R is None:
            R = np.eye(3)
        if t is None:
            t = np.zeros(3)

        self.R = copy.deepcopy(R)  # 3*3 rotation matrix
        self.t = copy.deepcopy(np.reshape(t, (3, 1)))  # 3*1 position

    def homogeneous_matrix(self):
        """returns the 4*4 homogeneous matrix."""
        H = np.zeros((4, 4))
        H[0:3, 0:3] = self.R
        H[0:3, 3:4] = self.t
        H[3, 3] = 1
        return H

    @staticmethod
    def invert_homogeneous_matrix(H: np.ndarray) -> np.ndarray:
        """returns the inverse of a homegeneous matrix."""
        H_rm = RigidMotion(H=H)
        return RigidMotion.inverse(H_rm).homogeneous_matrix()

    @staticmethod
    def inverse(H):
        """Returns H^-1 given RigidMotion H"""
        inv_R = (H.R).T
        inv_t = -1.0 * (inv_R @ H.t)
        return RigidMotion(R=inv_R, t=inv_t)

    @staticmethod
    def exp_skew_symmetric(S: np.ndarray, t: float = 1.0, theta: float = None) -> np.ndarray:
        """Returns exp(t*S) of a 3*3 skew symmetric matrix to get a rotation matrix."""

        if (S ** 2).sum() < 1e-8:
            # S is all zero
            return np.eye(3)

        if theta is None:
            s = np.array([S[2, 1], S[0, 2], S[1, 0]])
            theta = np.sqrt(np.sum(s ** 2.0))

        angle = t * theta
        theta_square = theta * theta

        R = np.eye(3) + np.sin(angle) / theta * S + (1 - np.cos(angle)) / theta_square * (S @ S)
        return R

    @staticmethod
    def log_rotation(R: np.ndarray):
        """Return the log(R) where R is a rotation matrix.  So it returns a skew symmetric matrix."""
        arg = 0.5 * (R[0, 0] + R[1, 1] + R[2, 2] - 1)  # in [-1, 1]
        if arg > -1:
            if arg < 1:
                # 0 < angle < pi
                angle = np.arccos(arg)
                sinAngle = np.sin(angle)
                c = 0.5 * angle / sinAngle
                S = c * (R - R.T)
            else:
                # arg == 1, angle == 0
                # R is the identity matrix and S is the zero matrix
                S = np.zeros([3, 3])
        else:  # arg == -1, angle == pi
            # R + I is symmetric.  To avoid bias, we use (R[i,j]+R[j,i]) / 2 for off-diagonal entries rather than R[i,j]
            s = np.zeros((3, 1))
            if R[0, 0] >= R[1, 1]:
                if R[0, 0] >= R[2, 2]:
                    # R[0,0] is the maximum diagonal term
                    s[0] = R[0, 0] + 1
                    s[1] = 0.5 * (R[0, 1] + R[1, 0])
                    s[2] = 0.5 * (R[0, 2] + R[2, 0])
                else:
                    # R[2,2] is the maximum diagonal term
                    s[0] = 0.5 * (R[2, 0] + R[0, 2])
                    s[1] = 0.5 * (R[2, 1] + R[1, 2])
                    s[2] = R[2, 2] + 1
            else:
                if R[1, 1] >= R[2, 2]:
                    # R[1,1] is the maximum diagonal term
                    s[0] = 0.5 * (R[1, 0] + R[0, 1])
                    s[1] = R[1, 1] + 1
                    s[2] = 0.5 * (R[1, 2] + R[2, 1])
                else:
                    # R[2,2] is the maximum diagonal term
                    s[0] = 0.5 * (R[2, 0] + R[0, 2])
                    s[1] = 0.5 * (R[2, 1] + R[1, 2])
                    s[2] = R[2, 2] + 1
            length = np.sqrt(np.sum(s ** 2.0))
            if length > 0:
                adjust = np.pi * np.sqrt(0.5) / length
                s = s * adjust
            else:
                s = s * 0

            S = get_cross_product_matrix(s)
            # S = np.array([
            #     [0, -s[2], s[1]],
            #     [s[2], 0, -s[0]],
            #     [-s[1], s[0], 0],
            # ])

        return S

    def rotate_translate(self, R: np.ndarray, t: np.ndarray):
        self.R = R @ self.R  # 3*3 rotation matrix
        self.t = self.t + np.reshape(t, (3, 1))  # 3*1 position
        return self

    @staticmethod
    def get_t_times_V(t: float, S: np.ndarray, theta: float = None):
        """
        Returns t * V(t, S), where
        V(t, S) = I + (1 - cos(t * theta)) / (t * theta)^2 * t * S + (t * theta - sin(t * theta)) / (t * theta)^3 * t^2 * S^2

        Args:
            t: scalar
            S: a skew symmetric matrix
            theta: can be None
        """
        if theta is None:
            s = np.array([S[2, 1], S[0, 2], S[1, 0]])
            theta = np.sqrt(np.sum(s ** 2.0))

        if theta > 0:
            angle = t * theta
            theta_square = theta * theta
            theta_cubic = theta * theta_square
            c0 = (1 - np.cos(angle)) / theta_square
            c1 = (angle - np.sin(angle)) / theta_cubic
            return t * np.eye(3) + c0 * S + c1 * (S @ S)
        else:
            return t * np.eye(3)

    @staticmethod
    def get_inv_V(S: np.ndarray, theta: float = None):
        """Return V^-1 = I - 0.5 * S + 1/theta^2 * (1 - theta * sin(theta) / (2 * (1 - cos(theta)))) * S^2"""

        if theta is None:
            s = np.array([S[2, 1], S[0, 2], S[1, 0]])
            theta = np.sqrt(np.sum(s ** 2.0))

        if theta > 0:
            theta_square = theta * theta
            c = 1 - (theta * np.sin(theta)) / (2 * (1 - np.cos(theta)))
            return np.eye(3) - 0.5 * S + (c / theta_square) * (S @ S)
        else:
            return np.eye(3)

    @staticmethod
    def multiply(H0, H1):
        """returns H0 * H1 in RigidMotion"""
        HM = H0.homogeneous_matrix() @ H1.homogeneous_matrix()
        return RigidMotion(R=HM[:3, :3], t=HM[:3, 3])

    @staticmethod
    def interp(t: float, H0: 'RigidMotion', H1: 'RigidMotion') -> 'RigidMotion':
        """
        Geodestic interpolation between RigidMotion H0 and RigidMotion H1.  t=0 -> H0, t=1 -> H1.
        """

        if not np.allclose(H0.R @ H0.R.T, np.eye(3)):
            warnings.warn('support only rigit transformation')
        if not np.allclose(H1.R @ H1.R.T, np.eye(3)):
            warnings.warn('support only rigit transformation')

        H0_inv = RigidMotion.inverse(H0)
        H = RigidMotion.multiply(H1, H0_inv)
        S = RigidMotion.log_rotation(H.R)
        s = np.array([S[2, 1], S[0, 2], S[1, 0]])
        theta = np.sqrt(np.sum(s ** 2.0))
        inv_V1 = RigidMotion.get_inv_V(S, theta)
        U = inv_V1 @ H.t

        interp_R = RigidMotion.exp_skew_symmetric(S, t, theta)
        interp_t_times_V = RigidMotion.get_t_times_V(t, S, theta)
        out_R = interp_R @ H0.R
        out_t = interp_R @ H0.t + interp_t_times_V @ U
        return RigidMotion(R=out_R, t=out_t)


def interp_homegeneous_matrices(
        t: float,
        H0: np.ndarray,
        H1: np.ndarray,
) -> np.ndarray:
    """
    Interpolate rotation and translation so that it
    follows the shortest path and has constant speed.

    Ref: https://www.geometrictools.com/Documentation/InterpolationRigidMotions.pdf

    Args:
        t:
            the interpolation weight in [0, 1].
            If 0, we return the camera pose of H0, and if 1, we return the pose of H1.
        H0:
            (4, 4)
        H1:
            (4, 4)

        Returns:
            (4, 4)

    Returns:
        4*4 homogeneous matrix (from camera cood to world coord)
    """

    H0_rm = RigidMotion(R=H0[:3, :3], t=H0[:3, 3])
    H1_rm = RigidMotion(R=H1[:3, :3], t=H1[:3, 3])
    H_rm = RigidMotion.interp(t, H0_rm, H1_rm)
    return H_rm.homogeneous_matrix()


def interp_homegeneous_tensors(
        t: T.Union[float, torch.Tensor],
        H0: torch.Tensor,
        H1: torch.Tensor,
) -> torch.Tensor:
    """
    Geodestic interpolation between RigidMotion H0 and RigidMotion H1.  t=0 -> H0, t=1 -> H1.

    Args:
        t:
            float or (*,), interpolation weight between 0 and 1, where
            0 we will return H0 and 1 will return H1.
        H0:
            (*, 4, 4)
        H1:
            (*, 4, 4) same size as H0.

    Returns:
        (*, 4, 4) same size as H0.

    """
    H0_inv = inv_homogeneous_tensors(H0)  # (*, 4, 4)
    H = H1 @ H0_inv
    S = RigidMotion.log_rotation(H.R)
    s = np.array([S[2, 1], S[0, 2], S[1, 0]])
    theta = np.sqrt(np.sum(s ** 2.0))
    inv_V1 = RigidMotion.get_inv_V(S, theta)
    U = inv_V1 @ H.t

    interp_R = RigidMotion.exp_skew_symmetric(S, t, theta)
    interp_t_times_V = RigidMotion.get_t_times_V(t, S, theta)
    out_R = interp_R @ H0.R
    out_t = interp_R @ H0.t + interp_t_times_V @ U
    return RigidMotion(R=out_R, t=out_t)


def get_min_R(
        v1: T.Union[np.ndarray, torch.Tensor],
        v2: T.Union[np.ndarray, torch.Tensor],
) -> T.Union[np.ndarray, torch.Tensor]:
    """
    Return the rotation matrix that rotates v1 to v2 in a geodestic manner.

    Args:
        v1: (*, 3) direction vector (unit norm),
        v2: (*, 3) direction vector (unit norm)

    Returns:
        (*, 3, 3) rotation matrix (s.t. v2 = R @ v1)
    """
    is_numpy = False
    if isinstance(v1, np.ndarray):
        is_numpy = True
        v1 = torch.from_numpy(v1)
    if isinstance(v2, np.ndarray):
        is_numpy = True
        v2 = torch.from_numpy(v2)

    v1_norm = torch.linalg.vector_norm(v1, ord=2, dim=-1)
    v2_norm = torch.linalg.vector_norm(v2, ord=2, dim=-1)
    assert torch.allclose(v1_norm, torch.ones(1, dtype=v1_norm.dtype, device=v1.device))
    assert torch.allclose(v2_norm, torch.ones(1, dtype=v2_norm.dtype, device=v2.device))

    k = torch.cross(v1, v2, dim=-1)  # (*, 3)
    cos_theta = (v1 * v2).sum(-1)  # (*,)

    *b_shape, d = v1.shape
    eye3 = torch.eye(3, device=v1.device, dtype=v1.dtype).expand(*b_shape, 3, 3)  # (*, 3, 3)
    if cos_theta > -1:
        Kx = get_cross_product_matrix(k)  # (*, 3, 3)
        R = eye3 + Kx + (Kx @ Kx) / (1 + cos_theta).unsqueeze(-1).unsqueeze(-1)  # (*, 3, 3)
    else:
        # v1 and v2 point to opposite directions
        R = -1 * eye3

    if is_numpy:
        R = R.detach().cpu().numpy()

    return R  # (*, 3, 3)


def get_cross_product_matrix(
        v: T.Union[np.ndarray, torch.Tensor],
) -> T.Union[np.ndarray, torch.Tensor]:
    """
    Given a (*, 3) vector v, return a (*, 3, 3) cross_product matrix [v]_x,
    such that for all (3,) vector u, we have v * u = [v]_x * u.

    Vx = np.array([
        [0,    -v[2],  v[1]],
        [v[2],   0,   -v[0]],
        [-v[1], v[0],    0],
    ])

    """
    is_numpy = False
    if isinstance(v, np.ndarray):
        is_numpy = True
        v = torch.from_numpy(v)

    *b_shape, d = v.shape
    assert d == 3
    Vx = torch.zeros(*b_shape, 3, 3, dtype=v.dtype, device=v.device)
    Vx[..., 0, 1] = -v[..., 2]
    Vx[..., 0, 2] = v[..., 1]
    Vx[..., 1, 2] = -v[..., 0]
    Vx = Vx - Vx.transpose(-1, -2)

    if is_numpy:
        Vx = Vx.detach().cpu().numpy()

    return Vx  # (*, 3, 3)


def get_random_direction(*shape, rng: np.random.RandomState = None):
    """
    Return a random unit direction vector.

    If shape = (n1, n2), the function returns (n1, n2, 3).
    """

    if len(shape) == 0:
        shape = []

    if rng is None:
        vs = np.random.randn(*shape, 3)
    else:
        vs = rng.randn(*shape, 3)
    vs = vs / np.linalg.norm(vs, axis=-1, keepdims=True)

    return vs


def get_random_direction_within_cone(
        n: int,
        theta: float,
        rng: np.random.RandomState = None,
        method: str = 'random',
):
    """
    Get uniformly sampled random directions within a cone centered at (0,0,1).

    Args:
        n:
            number of samples
        theta:
            the half angle of the cone (in degree)
        rng:
            numpy random state
        method:
            'random'

    Returns:
        (n, 3), float64

    We are to use Archimedes' Hat-Box Theorem to sample the directions.
    """
    assert 0 < theta <= 180.
    t_max = 1
    t_min = np.cos(theta / 180. * np.pi)

    if rng is None:
        rng = np.random

    # sample uniformly within [0,1]^2
    samples = sample_utils.get_samples(total_samples=n, d=2, method=method, rng=rng)  # (n, 2)
    # adjust range
    wzs = samples[..., 0] * (t_max - t_min) + t_min  # (n,)
    phis = samples[..., 1] * (2 * np.pi)  # (n,)

    rs = np.sqrt(1 - np.power(wzs, 2))
    wxs = rs * np.cos(phis)
    wys = rs * np.sin(phis)

    ds = np.stack((wxs, wys, wzs), axis=-1)  # (n,3)
    return ds


def construct_coord_frame(
        z: T.Union[np.ndarray, torch.Tensor] = (0, 0, -1.),
        y: T.Union[np.ndarray, torch.Tensor] = (0, -1., 0.),
) -> T.Union[np.ndarray, torch.Tensor]:
    """
    Get a coordinate frame from z and y vector.
    z will be used directly as the z axis.
    y will be made orthogonal to y and used as the y axis.
    x axis will be the the cross-product of y axis and z axis.
    All axes are normalised to have unit norm.

    Args:
        z: (*, 3)
        y: (*, 3)

    Returns:
        (*, 3, 3):
        For the last 2 dimension, the first column is the x axis, second y, last z.
        It can be used as the rotation matrix that transform
        a vector in camera coord to world coord.
    """

    if isinstance(z, (tuple, list)):
        z = torch.tensor(z)
    if isinstance(y, (tuple, list)):
        y = torch.tensor(y)

    is_numpy = False
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z)
        is_numpy = True
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
        is_numpy = True

    z_norm = torch.linalg.norm(z, ord=2, dim=-1, keepdim=True)  # (*, 1)
    assert torch.all(z_norm > 0)
    assert torch.all(torch.linalg.norm(y, ord=2, dim=-1) > 0)
    x = torch.cross(y, z, dim=-1)  # (*, 3)
    if torch.any(torch.linalg.norm(x, ord=2, dim=-1) == 0):
        raise ValueError("y and z cannot be parallel.")

    # make sure y-axis is perpendicular to z-axis
    z = z / z_norm  # (*, 3)
    y_on_z = torch.sum(y * z, dim=-1, keepdim=True) * z  # (*, 3)
    y = y - y_on_z

    # normalize
    y = y / torch.linalg.norm(y, ord=2, dim=-1, keepdim=True)  # (*, 3)
    x = x / torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)  # (*, 3)

    Rs = torch.stack((x, y, z), dim=-1)  # (*, 3, 3)

    if is_numpy:
        Rs = Rs.detach().cpu().numpy()

    return Rs


def get_H_c2w_lookat(
        pinhole_location_w: T.Union[np.ndarray, torch.Tensor] = (0, 0, 0.),
        look_at_w: T.Union[np.ndarray, torch.Tensor] = (0, 0, -1.),
        up_w: T.Union[np.ndarray, torch.Tensor] = (0, 1, 1.),
        invert_y: bool = True,
) -> T.Union[np.ndarray, torch.Tensor]:
    """
    Construct a camera pose homogeneous matrix H_c2w.

    Args:
        pinhole_location_w:
            (*, 3) pinhole location in the world coordinate
        look_at:
            (*, 3) a point in the world coordinate the optical axis of the camera will pass through
            should not be pinhole_location_w
        up:
            (*, 3) a vector roughly pointing upward in the world coordinate
            no need to normalize to have unit norm
        invert_y:
            whether to invert the y axis (since image coordinate is x to right y to down)

    Returns:
        (*, 4, 4)  H_c2w
    """
    if isinstance(pinhole_location_w, (tuple, list)):
        pinhole_location_w = torch.tensor(pinhole_location_w)
    if isinstance(look_at_w, (tuple, list)):
        look_at_w = torch.tensor(look_at_w)
    if isinstance(up_w, (tuple, list)):
        up_w = torch.tensor(up_w)

    is_numpy = False
    if isinstance(pinhole_location_w, np.ndarray):
        pinhole_location_w = torch.from_numpy(pinhole_location_w)
        is_numpy = True
    if isinstance(look_at_w, np.ndarray):
        look_at_w = torch.from_numpy(look_at_w)
        is_numpy = True
    if isinstance(up_w, np.ndarray):
        up_w = torch.from_numpy(up_w)
        is_numpy = True

    # construct coordinate frame of the camera (note we flip y-axis by default)
    cam_coords = construct_coord_frame(
        z=look_at_w - pinhole_location_w,
        y=-up_w if invert_y else up_w,
    )  # (*, 3, 3)

    *b_shape, a, b = cam_coords.shape
    H_c2w = torch.zeros(*b_shape, 4, 4, device=pinhole_location_w.device)
    H_c2w[..., :3, :3] = cam_coords
    H_c2w[..., :3, 3] = pinhole_location_w
    H_c2w[..., 3, 3] = 1

    if is_numpy:
        H_c2w = H_c2w.detach().cpu().numpy()
    return H_c2w


def get_H_c2w_Rt(
        R_w: T.Union[np.ndarray, torch.Tensor],
        t_w: T.Union[np.ndarray, torch.Tensor],
        invert_y: bool = True,
) -> T.Union[np.ndarray, torch.Tensor]:
    """
    Construct a camera pose homogeneous matrix H_c2w from camera frame R_w and position t_w.

    Args:
        R_w:
            (*, 3, 3) camera coordinate frame in the world coordinate (before flipping y-axis)
        t_w:
            (*, 3) pinhole position in the world coordinate (before flipping y-axis)
        invert_y:
            whether to invert the y axis (since image coordinate is x to right y to down)

    Returns:
        (*, 4, 4)  H_c2w
    """

    is_numpy = False
    if isinstance(R_w, np.ndarray):
        R_w = torch.from_numpy(R_w)
        is_numpy = True
    if isinstance(t_w, np.ndarray):
        t_w = torch.from_numpy(t_w)
        is_numpy = True

    # construct coordinate frame of the camera (note we flip y-axis by default)
    if invert_y:
        R_w[..., 1] = R_w[..., 1] * -1

    *b_shape, _, _ = R_w
    H_c2w = torch.zeros(*b_shape, 4, 4, device=R_w.device)
    H_c2w[..., :3, :3] = R_w
    H_c2w[..., :3, 3] = t_w
    H_c2w[..., 3, 3] = 1

    if is_numpy:
        H_c2w = H_c2w.detach().cpu().numpy()
    return H_c2w


def generate_random_camera_poses(
        n: int,
        max_angle: float,
        min_r: float,
        max_r: float,
        center_direction_w: T.List[float] = None,
        rng: np.random.RandomState = None,
        local_max_angle: float = 0.,
        rand_r: float = 0,
        origin_w: T.List[float] = None,  # (0., 0., 0.,),
        dtype: np.dtype = np.float32,
        method: str = 'random',
) -> T.List[np.ndarray]:
    """
    Generate `n` random camera poses, all of them within a cone of angle of `max_angle` (in degree)
    (i.e., -max_angle, max_angle) pointing toward a random direction.
    The camera centers to the origin are within min_r to max_r.

    If `local_max_angle` = 0, the camera will point to a random point with a sphere with r=rand_r.

    Args:
        max_angle:
            in degree. within the range of [0, 180.],  if 180, covers the entire sphere.
        center_direction_w:
            the center direction of the cone. If None, randomly choose one.
        local_max_angle:
            A small random xyz rotation (in degree) for each camera
        rand_r:
            a box with radius `center_r` within which the camera will be looking at
        origin_w: 
            the origin in the world coordinate where the camera is randomly placed.
            If None, origin_w = (0, 0, 0,)
        method:
            'random'
    """

    np_dtype = sample_utils.get_np_dtype(dtype)
    torch_dtype = sample_utils.get_np_dtype(dtype)

    if rng is None:
        rng = np.random

    # we first assume the world coordinate's y axis and z axis are flipped.
    # in order words, we have x to right, y to up, z to us.
    # We will deal with it at the end.

    if center_direction_w is None:
        # randomly sample a center direction
        d0 = get_random_direction(rng=rng)  # (3,)  float64
    else:
        d0 = np.array(center_direction_w, dtype=np.float64)  # (3,)  float64

    # randomly sample within the cone centered at (0,0,1) and spanning theta degrees each direction
    ds = get_random_direction_within_cone(
        n=n,
        theta=max_angle,
        rng=rng,
        method=method,
    )  # (n, 3) float64

    # rotate the directions to center at d0
    R = get_min_R(v1=np.array([0, 0, 1.]), v2=d0)  # float64
    ds = ds @ R.T  # (n,3)

    # since y axis is assumed flipped, we will use (0,1,0) as up
    y = np.array([0, 1., 0.])
    cam_frames = [construct_coord_frame(z=ds[i], y=y) for i in range(n)]  # list of 3*3 frame

    # randomly sample the camera pinhole distance to the origin
    rs = sample_utils.get_samples(total_samples=n, d=1, method=method, rng=rng, shuffle=True)  # (n, 1)
    rs = rs[:, 0] * (max_r - min_r) + min_r  # (n, )
    # rs = rng.rand(n) * (max_r - min_r) + min_r  # (n,)

    # the camera pinhole is at -d * r (so it looks at the origin)
    ts = -ds * np.expand_dims(rs, axis=1)  # (n, 3)

    # shift where the camera is looking at from the origin to a random
    # point within a box of radius rand_r
    if np.fabs(rand_r) > 1e-6:
        rts = sample_utils.get_samples(total_samples=n, d=3, method=method, rng=rng, shuffle=True)  # (n, 3)
        # rts = rng.rand(n, 3)
        ts = ts + (rts - 0.5) * 2 * rand_r  # (n, 3)
        # make sure ts is not within min_r
        ts_r = np.linalg.norm(ts, ord=2, axis=-1, keepdims=True)  # (n, 1)
        ts_r_clipped = np.clip(ts_r, a_min=min_r, a_max=max_r)
        ts = ts / ts_r * ts_r_clipped

    # construct the homogeneous matrix
    Hs = [RigidMotion(R=cam_frames[i], t=ts[i]).homogeneous_matrix() for i in range(n)]

    # invert y and z axis, since image coordinate we have x to right, y to down, z to far.
    H = np.eye(4)
    H[1, 1] = -1.
    H[2, 2] = -1.
    Hs = [H @ Hs[i] for i in range(n)]

    if local_max_angle > 1e-6:
        # generate a small random rotation for each camera
        rrs = sample_utils.get_samples(total_samples=n, d=3, method=method, rng=rng, shuffle=True)  # (n, 3)
        rs = (rrs - 0.5) * 2 * local_max_angle  # (n,3) random xyz rotation
        Rs = Rotation.from_euler('xyz', rs, degrees=True)
        H_locals = [RigidMotion(R=R.as_matrix()).homogeneous_matrix() for R in Rs]
        Hs = [H @ Hl for H, Hl in zip(Hs, H_locals)]

    Hs = [H.astype(np_dtype) for H in Hs]

    # translate the cameras by origin_w
    if origin_w is not None:
        origin_w = np.array(origin_w, dtype=np_dtype)
        for i in range(len(Hs)):
            Hs[i][:3, 3] = Hs[i][:3, 3] + origin_w

    return Hs


def inv_homogeneous_tensors(Hs: torch.Tensor):
    """
    Compute the inverse of the homogeneous matrices.

    Args:
        Hs: (*, 4, 4)

    Returns:
        inv_Hs: (*, 4, 4)
    """

    inv_Hs = torch.zeros_like(Hs)
    inv_Rs = Hs[..., :3, :3].transpose(-2, -1)
    inv_Hs[..., :3, :3] = inv_Rs
    inv_Hs[..., :3, 3:4] = -1.0 * (inv_Rs @ Hs[..., :3, 3:4])
    inv_Hs[..., 3, 3] = 1

    return inv_Hs
