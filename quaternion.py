from math import sqrt
import numpy as np

class Quaternion:
    def __init__(self, s=0, i=0, j=0, k=0):
        self.v = np.array([s,i,j,k])

    def __eq__(self, other):
        return np.array_equal(self.v, other.v)

    def __mul__(self, other):
        s = sum(self.v * other.v * np.array([1, -1, -1, -1]))
        i = sum(self.v * np.array([other.v[1], other.v[0], other.v[3], -other.v[2]]))
        j = sum(self.v * np.array([other.v[2], -other.v[3], other.v[0], other.v[1]]))
        k = sum(self.v * np.array([other.v[3], other.v[2], -other.v[1], other.v[0]]))

        return Quaternion(s, i, j, k)

    def L2Norm(self):
        return sqrt(sum(self.v**2))

    def L2Normalize(self):
        norm = self.L2Norm()
        self.v = self.v / norm

    def ToRotationMatrix(self):
        r = np.empty([3,3])
        w = self.v[0]
        x = self.v[1]
        y = self.v[2]
        z = self.v[3]
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
