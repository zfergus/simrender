import numpy as np
from scipy.spatial.transform import Rotation as R


def homogeneous(a):
    assert(len(a) == 3)
    return np.array([*a, 1.0], dtype=float)


def scale(s):
    if isinstance(s, (int, float)):
        return np.diag([s, s, s, 1.0])
    assert(len(s) == 3)
    return np.diag([*s, 1.0])


def translate(t):
    T = np.eye(4, dtype=float)
    T[:3, 3] = t
    return T


def align_vectors(a, b):
    """Create a rotation matrix that aligns vector a to vector b."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.align_vectors(
        np.array(b).reshape(1, -1),
        np.array(a).reshape(1, -1)
    )[0].as_matrix()
    return T
