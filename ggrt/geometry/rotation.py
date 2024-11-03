import math
import numpy as np


class Quaternion(object):
    def __init__(self, quat: np.ndarray) -> None:
        self.quat = quat

    @classmethod
    def normalize(cls, quat: np.ndarray) -> np.ndarray:
        """
        Args:
            quat: quaternion vector in [qw, qx, qy, qz]
        Return:
            The normalized quaternion such that 
                qw * qw + qx * qx + qy * qy + qz * qz = 1.
        """
        # qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        norm = np.linalg.norm(quat)
        normalized_quat = quat[:] / norm
        return normalized_quat

    @classmethod
    def inverse(cls, quat: np.ndarray) -> np.ndarray:
        """
        Args:
            quat: quaternion vector in [qw, qx, qy, qz]
        Return:
            The quaternion in the opposite direction.
        """
        quat = Quaternion.normalize(quat)
        quat[1:] = -quat[1:]
        return quat

    @classmethod
    def to_rotation_matrix(cls, quat: np.ndarray) -> np.ndarray:
        """
        Args:
            quat: quaternion vector in [qw, qx, qy, qz]
        Return:
            The 3*3 rotation matrix.
        """
        quat = Quaternion.normalize(quat)
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        R = np.zeros((3, 3), dtype=quat.dtype)
        R[0, 0] = qw ** 2 + qx ** 2 - qy ** 2 - qz ** 2
        R[0, 1] = 2. * (qx * qy - qw * qz)
        R[0, 2] = 2. * (qx * qz + qw * qy)
        R[1, 0] = 2. * (qx * qy + qw * qz)
        R[1, 1] = qw ** 2 - qx ** 2 + qy ** 2 - qz ** 2
        R[1, 2] = 2. * (qy * qz - qw * qx)
        R[2, 0] = 2. * (qx * qz - qw * qy)
        R[2, 1] = 2. * (qy * qz + qw * qx)
        R[2, 2] = qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2

        return R

    @classmethod
    def to_angle_axis(cls, quat: np.ndarray) -> np.ndarray:
        # NOTE: quaternion should be normalized!
        quat = Quaternion.normalize(quat)
        angle_axis = np.zeros(3)

        qx, qy, qz = quat[1], quat[2], quat[3]
        sin_squared_theta = qx * qx + qy * qy + qz * qz

        # For quaternions representing non-zero rotation, the conversion
        # is numerically stable.
        if sin_squared_theta > 0.0:
            sin_theta = np.sqrt(sin_squared_theta)
            cos_theta = quat[0]

            # If cos_theta is negative, theta is greater than pi/2, which means
            # that angle for the angle_axis vector which is 2*theta would be greater 
            # than pi.
            #
            # While this will result in the correct rotation, it does not result in 
            # a normalized angle-axis vector.
            # In that case we observe that 2*theta ~ 2*theta - 2*pi, which is 
            # equivalent saying: 
            #
            # theta - pi = atan(sin(theta - pi), cos(theta - pi))
            #            = atan(-sin(theta), -cos(theta))
            # 
            two_theta = 2. * np.arctan2(-sin_theta, -cos_theta) \
                        if cos_theta < 0. else 2. * np.arctan2(sin_theta, cos_theta)
            k = two_theta / sin_theta
            angle_axis[0], angle_axis[1], angle_axis[2] = k * qx, k * qy, k * qz
        else:
            # For zero rotation, sqrt() will produce NaN in the derivative since 
            # the argument is zero. By approximating with a Taylor series, and 
            # truncating at one term, the value and first derivatives will be computed
            # correctly.
            k = 2.
            angle_axis[0], angle_axis[1], angle_axis[2] = k * qx, k * qy, k * qz
        
        return angle_axis
    
    def rotate_point(self, point3D):
        pass


class AngleAxis(object):
    def __init__(self, angle_axis: np.ndarray) -> None:
        self.rotation_vec = angle_axis
    
    # @classmethod
    # def normalize(cls, angle_axis: np.ndarray):
    #     norm = np.linalg.norm(angle_axis)
    #     normalized_angle_axis = angle_axis[:] / norm
    #     return normalized_angle_axis

    @classmethod
    def to_rotation_matrix(cls, angle_axis: np.ndarray):
        """
        """
        a0, a1, a2 = angle_axis[0], angle_axis[1], angle_axis[2]
        theta_squared = a0 ** 2 + a1 ** 2 + a2 ** 2
        R = np.zeros((3, 3), dtype=angle_axis.dtype)
        min_threshold = 1e-15

        # We need to be careful to only evaluate the square root if the norm of the
        # rotation vector is greater than zero. Otherwise, we get a division by zero.
        if theta_squared > min_threshold:
            theta = np.sqrt(theta_squared)
            wx, wy, wz = a0 / theta, a1 / theta, a2 / theta

            sin_theta, cos_theta = np.sin(theta), np.cos(theta)

            # The equation is derived from the Rodrigues formula.
            R[0, 0] = cos_theta + wx * wx * (1. - cos_theta)
            R[1, 0] = wz * sin_theta + wx * wy * (1. - cos_theta)
            R[2, 0] = -wy * sin_theta + wx * wz * (1. - cos_theta)
            R[0, 1] = wx * wy * (1. - cos_theta) - wz * sin_theta
            R[1, 1] = cos_theta + wy * wy * (1. - cos_theta)
            R[2, 1] = wx * sin_theta + wy * wz * (1. - cos_theta)
            R[0, 2] = wy * sin_theta + wx * wz * (1. - cos_theta)
            R[1, 2] = -wx * sin_theta + wy * wz * (1. - cos_theta)
            R[2, 2] = cos_theta + wz * wz * (1. - cos_theta)
        else:
            # Near zero, we switch to using the first order Taylor expansion.
            R[0, 0], R[1, 0], R[2, 0] = 1., a2, -a1
            R[0, 1], R[1, 1], R[2, 1] = -a2, 1., a0
            R[0, 2], R[1, 2], R[2, 2] = a1, -a0, 1.

        return R

    @classmethod
    def to_quaternion(cls, angle_axis: np.ndarray):
        quat = np.zeros(4, dtype=angle_axis.dtype)

        a0, a1, a2 = angle_axis[0], angle_axis[1], angle_axis[2]
        # theta = np.linalg.norm(angle_axis)
        theta_squared = a0 ** 2 + a1 ** 2 + a2 ** 2

        # For points not at the origin, the full conversion is numerically stable.
        if theta_squared > 0.:
            theta = np.sqrt(theta_squared)
            half_theta = theta / 2
            k = np.sin(half_theta) / theta
            quat[0] = np.cos(half_theta)
            quat[1], quat[2], quat[3] = k * a0, k * a1, k * a2
        else:
            # At the origin,sqrt will produce NaN in the derivative since
            # the argument is zero. By approximating with a Taylor series,
            # and truncating at one term, the value and first derivatives
            # will be computed correctly.
            k = 0.5
            quat[0], quat[1], quat[2], quat[3] = 1., k * a0, k * a1, k * a2
        
        return quat

    @classmethod
    def theta(cls, angle_axis: np.ndarray):
        # angle_axis = AngleAxis.normalize(angle_axis)
        # a0, a1, a2 = angle_axis[0], angle_axis[1], angle_axis[2]
        return np.linalg.norm(angle_axis)

    def rotation_point(self, point3D):
        pass


class Rotation(object):
    def __init__(self, R: np.ndarray) -> None:
        self.R = R
    
    @classmethod
    def to_quaternion(cls, R):
        # Ref: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        quat = np.zeros(4, dtype=R.dtype)
        R00, R01, R02 = R[0, 0], R[0, 1], R[0, 2]
        R10, R11, R12 = R[1, 0], R[1, 1], R[1, 2]
        R20, R21, R22 = R[2, 0], R[2, 1], R[2, 2]
        trace = R00 + R11 + R22

        r = math.sqrt(1 + trace)
        qw = .5 * r
        qx = np.sign(R21 - R12) * .5 * math.sqrt(1. + R00 - R11 - R22)
        qy = np.sign(R02 - R20) * .5 * math.sqrt(1. - R00 + R11 - R22)
        qz = np.sign(R10 - R01) * .5 * math.sqrt(1. - R00 - R11 + R22)
        quat[0], quat[1], quat[2], quat[3] = qw, qx, qy, qz
        
        # If the matrix contains significant error, such as accumulated numerical 
        # error, we may construct a symmetric 4*4 matrix.
        if np.isnan(quat).any():
            K = np.array([[R00 - R11 - R22, R10 + R01, R20 + R02, R21 - R12],
                          [R10 + R01, R11 - R00 - R22, R21 + R12, R02 - R20],
                          [R20 + R02, R21 + R12, R22 - R00 - R11, R10 - R01],
                          [R21 - R12, R02 - R20, R10 - R01, R00 + R11 + R22]], dtype=R.dtype)
            eigen_values, eigen_vecs = np.linalg.eigh(K)
            qx, qy, qz, qw = eigen_vecs[eigen_values.argmax()]
            quat[0], quat[1], quat[2], quat[3] = qw, qx, qy, qz
            if quat[0] < 0:
              quat *= -1
        
        return quat

    @classmethod
    def to_angle_axis(cls, R):
        # We do not compute the angle axis by the Rodrigues formula.
        quat = Rotation.to_quaternion(R)
        angle_axis = Quaternion.to_angle_axis(quat)
        
        return angle_axis

    def transpose(self):
        return self.R.t()

    def chordal_distance(self, Q: np.ndarray):
        R_diff = self.R - Q
        return np.linalg.norm(R_diff)

    def angular_distance(self, Q: np.ndarray):
        relative_rotation = self.transpose() * Q
        angle_axis = Rotation.to_angle_axis(relative_rotation)
        return AngleAxis.theta(angle_axis)

    def rotate_point(self, point3D):
        return np.dot(self.R, point3D)

    def left_multiply(self, R: np.ndarray):
        self.R = self.R @ R

    def right_multiply(self, R: np.ndarray):
        self.R = R @ self.R


def euler_angle_to_rotation_matrix(euler_angles: np.ndarray):
    rx, ry, rz = euler_angles[0], euler_angles[1], euler_angles[2]
    Rx = np.array([[1.,           0.,            0.],
                   [0., math.cos(rx), -math.sin(rx)],
                   [0,  math.sin(rx),  math.cos(rx)]], dtype=euler_angles.dtype)
    Ry = np.array([[math.cos(ry),  0., math.sin(ry)],
                   [          0.,  1.,           0.],
                   [-math.sin(ry), 0., math.cos(ry)]], dtype=euler_angles.dtype)
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0.],
                   [math.sin(rz),  math.cos(rz), 0.],
                   [          0.,          0.,   1.]], dtype=euler_angles.dtype)

    return Rz @ Ry @ Rx


def rad_to_deg(angle):
    return angle * 180 / math.pi


def deg_to_rad(angle):
    return angle * math.pi / 180
