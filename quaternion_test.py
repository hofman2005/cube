from quaternion import Quaternion
import numpy as np
import numpy.testing as npt
import unittest

class QuaternionTest(unittest.TestCase):
    def test_Mul(self):
        a = Quaternion([1,2,3,4])
        b = Quaternion([2,3,4,5])
        c = Quaternion([-36, 6, 12, 12])
        self.assertEqual(c, a * b)
        pass

    def test_ToRotationMatrix(self):
        a = Quaternion([0.707, 0.707, 0, 0])
        r = a.ToRotationMatrix()
        npt.assert_almost_equal(r, np.array([[1,0,0],[0,0,-1],[0,1,0]]),
                decimal = 3)

        a = Quaternion([1, 0, 0, 90], False)
        r = a.ToRotationMatrix()
        npt.assert_almost_equal(r, np.array([[1,0,0],[0,0,-1],[0,1,0]]),
                decimal = 3)

if __name__ == "__main__":
    unittest.main()
