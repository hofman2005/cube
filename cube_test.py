import unittest
import cube
import numpy as np

class TestCube2x2(unittest.TestCase):
    def setUp(self):
        self.cube2x2 = cube.Cube2x2()

    def test_init(self):
        np.testing.assert_array_equal(self.cube2x2.cells[0].pos,
                                      np.array([-1,-1,-1]))

    def test_findLayer(self):
        layer = self.cube2x2.findLayer(x=1)

    def test_Rotate(self):
        print(self.cube2x2.GetObservation())
        self.cube2x2.rotate('R')
        print(self.cube2x2.GetObservation())
        self.cube2x2.rotate('r')
        self.cube2x2.rotate('F')
        self.cube2x2.rotate('f')
        self.cube2x2.rotate('U')
        self.cube2x2.rotate('u')
        print(self.cube2x2.GetObservation())

    def test_IsDone(self):
        self.assertTrue(self.cube2x2.IsDone())
        self.cube2x2.rotate('R')
        self.assertFalse(self.cube2x2.IsDone())
        self.cube2x2.rotate('r')
        self.assertTrue(self.cube2x2.IsDone())


if __name__ == "__main__":
    unittest.main()
