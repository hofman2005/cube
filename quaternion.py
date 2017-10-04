from math import sqrt
import numpy as np

class Quaternion:
    def __init__(self, vec, is_raw=True):
        vec = np.array(vec)
        if is_raw:
            # Input is [w, x, y, z]
            self.v = vec
        else:
            # Input is [axis, angle]
            angle = vec[3] / 180.0 * np.pi
            cos_angle = np.cos(angle/2.0)
            sin_angle = np.sin(angle/2.0)
            s = np.sqrt(np.sum(vec[0:3]**2.0))
            x = vec[0] / s
            y = vec[1] / s
            z = vec[2] / s
            self.v = np.array([cos_angle, sin_angle * x, sin_angle * y,
                sin_angle * z])

    def __eq__(self, other):
        return np.array_equal(self.v, other.v)

    def __mul__(self, other):
        s = np.sum(self.v * other.v * np.array([1, -1, -1, -1]))
        i = np.sum(self.v * np.array([other.v[1], other.v[0], other.v[3], -other.v[2]]))
        j = np.sum(self.v * np.array([other.v[2], -other.v[3], other.v[0], other.v[1]]))
        k = np.sum(self.v * np.array([other.v[3], other.v[2], -other.v[1], other.v[0]]))

        return Quaternion([s, i, j, k])

    def L2Norm(self):
        return sqrt(sum(self.v**2))

    def L2Normalize(self):
        norm = self.L2Norm()
        self.v = self.v / norm

    def ToRotationMatrix(self):
        r = np.empty([3,3])
        s = np.sqrt(np.sum(self.v**2))
        w = self.v[0] / s
        x = self.v[1] / s
        y = self.v[2] / s
        z = self.v[3] / s
        r[0,0] = 1 - 2 * y ** 2 - 2 * z ** 2
        r[0,1] = 2 * x * y - 2 * z * w
        r[0,2] = 2 * x * z + 2 * y * w
        r[1,0] = 2 * x * y + 2 * z * w
        r[1,1] = 1 -2 * x ** 2 - 2 * z ** 2
        r[1,2] = 2 * y * z - 2 * x * w
        r[2,0] = 2 * x * z - 2 * y * w
        r[2,1] = 2 * y * z + 2 * x * w
        r[2,2] = 1 - 2 * x ** 2 - 2 * y ** 2
        return r
