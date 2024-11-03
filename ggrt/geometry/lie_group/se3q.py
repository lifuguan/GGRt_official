from math import pi
from typing import Dict, Union

import torch

from ggrt.geometry.lie_group import se3_common as se3c
from ggrt.geometry.lie_group.so3q import SO3q
from ggrt.geometry.lie_group.liegroupbase import LieGroupBase
from ggrt.geometry.lie_group.utils import isclose


class SE3q(LieGroupBase):

    DIM = 7
    DOF = 6
    N = 4  # Group transformation is 4x4 matrices
    name = 'SE3qTorch'

    @staticmethod
    def identity(size: int = None) -> 'SE3q':
        if size is None:
            return SE3q(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        else:
            vec = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).repeat(size, 1)
            return SE3q(vec)

    @staticmethod
    def _from_rt(rot: Union[SO3q, torch.Tensor], trans: torch.Tensor) -> 'SE3q':
        """Convenience function to concatenates the rotation and translation
        part into a SE(3) matrix

        Args:
            rot: ([*,] 4) or SO3q
            trans: ([*,] 3)

        Returns:
            SE(3) matrix
        """
        rotq = rot if isinstance(rot, torch.Tensor) else rot.data
        vec = torch.cat([rotq, trans], dim=-1)
        return SE3q(vec)

    @staticmethod
    def from_rtvec(vec: torch.Tensor) -> 'SE3q':
        """Constructs from 7D vector"""
        assert se3c.is_valid_quat_trans(vec)
        return SE3q(vec)

    @staticmethod
    def from_matrix(mat: torch.Tensor) -> 'SE3q':
        assert mat.shape[-2:] == (4, 4), 'Matrix should be of shape ([*,] 4, 4)'
        assert se3c.is_valid_matrix(mat), 'Matrix is not a valid SE(3)'

        rot = SO3q.from_matrix(mat[..., :3, :3])
        trans = mat[..., :3, 3]
        return SE3q._from_rt(rot, trans)

    def inv(self) -> 'SE3q':
        irot = SO3q(self.data[..., :4]).inv()
        trans = self.trans  # ([N, ] 3)
        itrans = -irot.transform(trans[..., None, :])[:, 0, :]
        return SE3q._from_rt(irot, itrans)

    def __mul__(self, other: 'SE3q') -> 'SE3q':
        """Quaternion multiplication.

        Computes qout = q1 * q2, where * is the Hamilton product between the two
        quaternions. Note that the Hamiltonian product is not commutative.
        """
        rot = self.rot * other.rot
        trans = self.rot.transform(other.data[..., None, 4:])[..., 0, :] + self.trans
        rtvec = torch.cat([rot.data, trans], dim=-1)
        return SE3q(rtvec)

    def transform(self, pts: torch.Tensor) -> torch.Tensor:
        assert len(self.shape) == pts.ndim - 2
        transformed = self.rot.transform(pts) + self.data[..., None, 4:]
        return transformed

    @staticmethod
    def exp(vec: torch.Tensor) -> 'SE3q':
        """Group exponential. Converts an element of tangent space (twist) to the
        corresponding element of the group SE(3).

        To be specific, computes expm(hat(vec)) with expm being the matrix
        exponential and hat() being the hat operator of SE(3).

        Args:
            vec: Twist vector ([N, ] 6)

        Returns:
            SE(3) matrix of size ([N, ] 7)

        Credits: Implementation is inspired by that in Sophus library
                 https://github.com/strasdat/Sophus/blob/master/sophus/se3.hpp
        """
        orig_shape = vec.shape
        if vec.ndim == 1:
            vec = vec[None, :]

        v, omega = vec[..., :3], vec[..., 3:]
        rot, theta = SO3q.exp_and_theta(omega)

        V = vec.new_zeros((*vec.shape[:-1], 3, 3))
        small_theta_mask = isclose(theta, 0.0)

        if torch.sum(small_theta_mask) > 0:
            V[small_theta_mask] = rot[small_theta_mask].as_matrix()
        if torch.sum(~small_theta_mask) > 0:
            mask = ~small_theta_mask
            Omega = SO3q.hat(omega[mask])
            Omega_sq = Omega @ Omega

            theta_masked = theta[mask, None, None]
            theta2, theta3 = theta_masked ** 2, theta_masked ** 3
            s, c = torch.sin(theta_masked), torch.cos(theta_masked)

            V[~small_theta_mask] = torch.eye(3, device=vec.device, dtype=vec.dtype) \
                                   - (c - 1.0) / theta2 * Omega \
                                   + (theta_masked - s) / theta3 * Omega_sq

        trans = V @ v[..., None]
        rtvec = SE3q._from_rt(rot, trans[..., 0])
        rtvec.data = torch.reshape(rtvec.data, (*orig_shape[:-1], 7))
        return rtvec

    def log(self) -> torch.Tensor:
        """Logarithm map.
        """
        raise NotImplementedError

    @staticmethod
    def hat(v: torch.Tensor):
        """hat-operator for SE(3)
        Specifically, it takes in the 6-vector representation (= twist) and returns
        the corresponding matrix representation of Lie algebra element.

        Args:
            v: Twist vector of size ([*,] 6). As with common convention, first 3
               elements denote translation.

        Returns:
            mat: se(3) element of size ([*,] 4, 4)
        """
        return se3c.hat(v)

    @staticmethod
    def vee(mat: torch.Tensor):
        """vee-operator for SE(3), i.e. inverse of hat() operator.

        Args:
            mat: ([*, ] 4, 4) matrix containing the 4x4-matrix lie algebra
                 representation. Omega must have the following structure:
                     |  0 -f  e  a |
                     |  f  0 -d  b |
                     | -e  d  0  c |
                     |  0  0  0  0 | .

        Returns:
            v: twist vector of size ([*,] 6)

        """
        return se3c.vee(mat)

    """Comparison functions"""
    def compare(self, other: 'SE3q') -> Dict:
        """Compares two SO3 instances, returning the rotation error in degrees
        Note that for the translation error, we compare the translation portion
        directly directly and not on the error term, to be consistent with
        "Learning Transformation Synchronization" (CVPR2019)
        """
        error = self * other.inv()
        e = {'rot_deg': SO3q.rotation_angle(error.rot) * 180 / pi,
             'trans': torch.norm(self.trans - other.trans, dim=-1)}
        return e

    """Conversion functions"""
    @property
    def rot(self) -> SO3q:
        return SO3q(self.data[..., :4])

    @property
    def trans(self) -> torch.Tensor:
        return self.data[..., 4:]

    def vec(self) -> torch.Tensor:
        """Returns the flattened representation"""
        return self.data

    def as_quat_trans(self):
        """Return the 7D representation (quaternion, translation)
        First 4 columns contain the quaternion, last 3 columns contain translation
        """
        return self.data

    def as_matrix(self) -> torch.tensor:
        return se3c.quattrans2mat(self.data)

    def is_valid(self) -> bool:
        """Check whether the data is valid, e.g. if the underlying SE(3)
        representation has a valid rotation"""
        return se3c.is_valid_quat_trans(self.data)

    def make_valid(self) -> 'SE3q':
        """Rectifies the data so that the representation is valid"""
        return SE3q(se3c.normalize_quat_trans(self.data))

    @property
    def shape(self):
        return self.data.shape[:-1]