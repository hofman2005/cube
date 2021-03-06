import numpy as np
from enum import Enum
from quaternion import Quaternion

class Color(Enum):
    R=0
    Y=1
    W=2
    B=3
    G=4
    O=5

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
        for key in self.rep.keys():
            value = self.rep[key]
            key = np.dot(np.array(key), rotation_matrix.transpose()).astype(int)
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

        self.RotationQuaternion = {}
        self.RotationMatrix = {}

        self.RotationQuaternion['R'] = Quaternion([1,0,0,-90], False)
        self.RotationQuaternion['r'] = Quaternion([1,0,0,90], False)
        self.RotationQuaternion['L'] = Quaternion([-1,0,0,-90], False)
        self.RotationQuaternion['l'] = Quaternion([-1,0,0,90], False)
        self.RotationQuaternion['F'] = Quaternion([0,-1,0,-90], False)
        self.RotationQuaternion['f'] = Quaternion([0,-1,0,90], False)
        self.RotationQuaternion['B'] = Quaternion([0,1,0,-90], False)
        self.RotationQuaternion['b'] = Quaternion([0,1,0,90], False)
        self.RotationQuaternion['U'] = Quaternion([0,0,1,-90], False)
        self.RotationQuaternion['u'] = Quaternion([0,0,1,90], False)
        self.RotationQuaternion['D'] = Quaternion([0,0,-1,-90], False)
        self.RotationQuaternion['d'] = Quaternion([0,0,-1,90], False)

        for key in self.RotationQuaternion.keys():
            self.RotationMatrix[key] = self.RotationQuaternion[key].ToRotationMatrix()

        self.actions = [
                'R', 'r',
                'L', 'l',
                'F', 'f',
                'B', 'b',
                'U', 'u',
                'D', 'd',
                ]

        self.action_to_id_map = {}
        for i in range(len(self.actions)):
            self.action_to_id_map[self.actions[i]] = i

        self.reverse_actions = {
            'R': 'r',
            'r': 'R',
            'L': 'l',
            'l': 'L',
            'F': 'f',
            'f': 'F',
            'B': 'b',
            'b': 'B',
            'U': 'u',
            'u': 'U',
            'D': 'd',
            'd': 'D',
            }

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
            obs.append(cube.GetColor((1,0,0)).value)
        for cube in self.findLayer(x=-1):
            obs.append(cube.GetColor((-1,0,0)).value)
        for cube in self.findLayer(y=1):
            obs.append(cube.GetColor((0,1,0)).value)
        for cube in self.findLayer(y=-1):
            obs.append(cube.GetColor((0,-1,0)).value)
        for cube in self.findLayer(z=1):
            obs.append(cube.GetColor((0,0,1)).value)
        for cube in self.findLayer(z=-1):
            obs.append(cube.GetColor((0,0,-1)).value)
        return obs

    def IsDone(self):
        if len(set([o.GetColor((1,0,0)) for o in self.findLayer(x=1)])) != 1:
            return False
        if len(set([o.GetColor((-1,0,0)) for o in self.findLayer(x=-1)])) != 1:
            return False
        if len(set([o.GetColor((0,1,0)) for o in self.findLayer(y=1)])) != 1:
            return False
        if len(set([o.GetColor((0,-1,0)) for o in self.findLayer(y=-1)])) != 1:
            return False
        if len(set([o.GetColor((0,0,1)) for o in self.findLayer(z=1)])) != 1:
            return False
        if len(set([o.GetColor((0,0,-1)) for o in self.findLayer(z=-1)])) != 1:
            return False
        return True

    def rotate(self, action):
        if action not in self.RotationQuaternion:
            return False

        if action in ('R', 'r'):
            layer = self.findLayer(x=1)
        if action in ('L', 'l'):
            layer = self.findLayer(x=-1)
        if action in ('F', 'f'):
            layer = self.findLayer(y=-1)
        if action in ('B', 'b'):
            layer = self.findLayer(y=1)
        if action in ('U', 'u'):
            layer = self.findLayer(z=1)
        if action in ('D', 'd'):
            layer = self.findLayer(z=-1)
        for cell in layer:
            cell.Rotate(self.RotationQuaternion[action],
                    self.RotationMatrix[action])
            self.posMap[tuple(cell.pos)] = cell

        return True

    def GetActions(self):
        return self.actions

    def GetNumOfActions(self):
        return len(self.actions)
    
    def GetReverseAction(self, action):
        return self.reverse_actions[action]

    def GetActionById(self, action_id):
        return self.actions[action_id]

    def GetIdForAction(self, action):
        return self.action_to_id_map[action]
