"""SO3 using quaternion representation
Note that for edge cases, we only use up to first order approximation
"""
from math import pi
from typing import Dict

import torch

from ggrt.geometry.lie_group import so3_common as so3c
from ggrt.geometry.lie_group.liegroupbase import LieGroupBase
from ggrt.geometry.lie_group.utils import isclose


class SO3q(LieGroupBase):
    DIM = 4
    DOF = 3
    N = 3  # Group transformation is 3x3 matrices
    name = 'SO3qTorch'

    @staticmethod
    def identity(size: int = None, dtype=None, device=None) -> 'SO3q':
        if size is None:
            return SO3q(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device))
        else:
            return SO3q(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device).repeat(size, 1))

    @staticmethod
    def from_quaternion(quat: torch.Tensor, normalize: bool = False) -> 'SO3q':
        if normalize:
            quat = quat / torch.norm(quat, dim=-1)
        return SO3q(quat)

    @staticmethod
    def from_matrix(mat: torch.Tensor) -> 'SO3q':
        return SO3q(so3c.rotmat2quat(mat))

    @staticmethod
    def exp(omega: torch.Tensor) -> 'SO3q':
        q, _ = SO3q.exp_and_theta(omega)
        return q

    @staticmethod
    def exp_and_theta(omega: torch.Tensor) -> ('SO3q', torch.Tensor):
        """Same as exp() but also returns theta (rotation angle in radians)

        This follows the implementation from ceres solver
        https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/rotation.h
        """
        orig_shape = omega.shape
        if omega.ndim == 1:
            omega = omega[None, :]

        theta = torch.norm(omega, dim=-1)  # ([N,] 1)
        zero_theta = isclose(theta, 0.0)
        quats = omega.new_empty(size=(*omega.shape[:-1], 4))

        # For small rotations, use taylor 1st order approximation
        num_zero = int(zero_theta.sum())
        if num_zero > 0:
            quats[zero_theta] = torch.cat([
                omega.new_ones(size=(num_zero, 1)),
                omega[zero_theta] * 0.5],
                dim=-1)

        # Standard conversion is numerical stable for non-zero rotations
        if num_zero < omega.shape[0]:
            mask = ~zero_theta
            theta_masked = theta[mask][:, None]  # (N, 1)
            half_theta_masked = 0.5 * theta_masked
            k = torch.sin(half_theta_masked) / theta_masked
            quats[mask] = \
                torch.cat([torch.cos(half_theta_masked), omega[mask] * k],
                          dim=-1)

        quats = torch.reshape(quats, (*orig_shape[:-1], 4))
        return SO3q(quats), theta

    def log(self) -> torch.Tensor:
        """Converts quaternion to angle axis
        This follows the implementation from ceres solver
        https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/rotation.h
        """
        quat = self.data
        if quat.ndim == 1:
            quat = quat[None, :]  # (N, 4)

        omegas = quat.new_empty(size=(*quat.shape[:-1], 3))
        sin_theta = torch.norm(quat[..., 1:], dim=-1)  # (N, )
        zero_theta = sin_theta == 0.0  # (N, )

        if torch.sum(~zero_theta) > 0:
            mask = ~zero_theta
            cos_theta_masked = quat[mask, 0]

            two_theta0 = 2.0 * torch.atan2(sin_theta[mask], cos_theta_masked)
            two_theta1 = 2.0 * torch.atan2(-sin_theta[mask], -cos_theta_masked)
            two_theta = torch.where(cos_theta_masked >= 0, two_theta0, two_theta1)

            k = two_theta / sin_theta
            omegas[mask] = quat[mask, 1:] * k[:, None]

        if torch.sum(zero_theta) > 0:
            # Taylor 1st order approximation
            omegas[zero_theta] = quat[zero_theta, 1:] * 2.0

        omegas = torch.reshape(omegas, (*self.data.shape[:-1], 3))
        return omegas

    def inv(self) -> 'SO3q':
        """Quaternion inverse, which is equivalent to its conjugate"""
        return SO3q(so3c.quat_inv(self.data))

    def __mul__(self, other: 'SO3q') -> 'SO3q':
        """Quaternion multiplication.

        Computes qout = q1 * q2, where * is the Hamilton product between the two
        quaternions. Note that the Hamiltonian product is not commutative.
        """
        return SO3q(so3c.quat_mul(self.data, other.data))

    def transform(self, pts: torch.Tensor) -> torch.Tensor:
        assert len(self.shape) == pts.ndim - 2
        transformed = so3c.quat_rot(self.data, pts)
        return transformed

    @staticmethod
    def hat(v: torch.Tensor):
        """Maps a vector to a 3x3 skew symmetric matrix."""
        return so3c.hat(v)

    @staticmethod
    def vee(mat: torch.Tensor):
        """Inverse of hat operator, i.e. transforms skew-symmetric matrix to
        3-vector
        """
        return so3c.vee(mat)

    """Comparison functions"""

    def rotation_angle(self) -> torch.Tensor:
        """Returns the rotation angle in radians"""
        sin_theta = torch.clamp_max(torch.norm(self.data[..., 1:], dim=-1), max=1.0)  # (N, )
        return torch.asin(sin_theta) * 2.0

    def compare(self, other: 'SO3q') -> Dict:
        """Compares two SO3 instances, returning the rotation error in degrees"""
        error = self * other.inv()
        e = {'rot_deg': SO3q.rotation_angle(error) * 180 / pi}
        return e

    """Conversion functions"""

    def vec(self) -> torch.Tensor:
        """Returns the flattened representation"""
        return self.data

    def as_quaternion(self) -> torch.Tensor:
        return self.data

    def as_matrix(self) -> torch.Tensor:
        return so3c.quat2rotmat(self.data)

    def is_valid(self) -> bool:
        return so3c.is_valid_quaternion(self.data)

    def make_valid(self):
        """Rectifies the data so that the representation is valid"""
        return SO3q(so3c.normalize_quaternion(self.data))

    @property
    def shape(self):
        return self.data.shape[:-1]
