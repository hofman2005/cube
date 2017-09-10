import unittest
import cube
import numpy as np

class TestCube2x2(unittest.TestCase):
    def setUp(self):
        self.cube2x2 = cube.Cube2x2()

    def test_init(self):
        np.testing.assert_array_equal(self.cube2x2.cells[0].pos,
                                      np.array([-1,-1,-1]))

if __name__ == "__main__":
    unittest.main()
