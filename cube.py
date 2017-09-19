import numpy as np
from quaternion import Quaternion

class Cell:
    def __init__(self, name_ = "", pos_ = [0,0,0], ori_ = Quaternion(1,0,0,0)):
        self.pos = np.array(pos_)
        self.ori = np.array(ori_)
        self.name = name_

class Cube(Cell):
    def __init__(self):
        pass

class Cube2x2(Cube):
    def __init__(self):
        self.posMap = {}
        self.cells = []
        for x in (-1,1):
            for y in (-1,1):
                for z in (-1,1):
                    pos = (x, y, z)
                    self.cells.append(Cell("", pos))
                    self.posMap[pos] = self.cells[len(self.cells)-1]

    def findLayer(self, x=None, y=None, z=None):
        layer = []
        for i in (-1,1):
            for j in (-1,1):
                if x is not None:
                    pos = (x, i, j)
                elif y is not None:
                    pos = (i, y, j)
                elif z is not None:
                    pos = (i, j, z)
                else: return
                if pos in self.posMap:
                    layer.append(self.posMap[pos])
        return layer

    def rotateR(self):
        layer = findLayer(x=1)
