import numpy as np
from enum import Enum
from quaternion import Quaternion

class Color(Enum):
    R=1
    Y=2
    W=3
    B=4
    G=5
    O=6

class Cell:
    def __init__(self, name_ = "", pos_ = [0,0,0], ori_ = Quaternion([1,0,0,0])):
        self.pos = np.array(pos_)
        self.ori = ori_
        self.name = name_
        self.rep = {}

    def Rotate(self, quaternion, rotation_matrix):
        self.pos = np.dot(self.pos, rotation_matrix.transpose())
        self.pos = np.rint(self.pos)
        self.ori = quaternion * self.ori

        newRep = {}
        rotationMatrix = self.ori.ToRotationMatrix().transpose()
        for key in self.rep.keys():
            value = self.rep[key]
            key = np.dot(np.array(key), rotationMatrix).astype(int)
            key = tuple(key)
            newRep[key] = value
        self.rep = newRep

    def AddColor(self, pos, color):
        self.rep[pos] = color

    def GetColor(self, pos):
        return self.rep[pos]

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

        self.Rq = Quaternion([1,0,0,-90], False)
        self.Rm = self.Rq.ToRotationMatrix()

        for cube in self.findLayer(x=1):
            cube.AddColor((1,0,0), Color.Y)
        for cube in self.findLayer(x=-1):
            cube.AddColor((-1,0,0), Color.W)
        for cube in self.findLayer(y=1):
            cube.AddColor((0,1,0), Color.O)
        for cube in self.findLayer(y=-1):
            cube.AddColor((0,-1,0), Color.R)
        for cube in self.findLayer(z=1):
            cube.AddColor((0,0,1), Color.B)
        for cube in self.findLayer(z=-1):
            cube.AddColor((0,0,-1), Color.G)

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

    def GetObservation(self):
        obs = []
        for cube in self.findLayer(x=1):
            obs.append(cube.GetColor((1,0,0)).name)
        for cube in self.findLayer(x=-1):
            obs.append(cube.GetColor((-1,0,0)).name)
        for cube in self.findLayer(y=1):
            obs.append(cube.GetColor((0,1,0)).name)
        for cube in self.findLayer(y=-1):
            obs.append(cube.GetColor((0,-1,0)).name)
        for cube in self.findLayer(z=1):
            obs.append(cube.GetColor((0,0,1)).name)
        for cube in self.findLayer(z=-1):
            obs.append(cube.GetColor((0,0,-1)).name)
        return obs

    def rotateR(self):
        layer = self.findLayer(x=1)
        for cell in layer:
            cell.Rotate(self.Rq, self.Rm)
            self.posMap[tuple(cell.pos)] = cell
