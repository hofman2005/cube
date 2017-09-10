import numpy as np

class Cell:
    def __init__(self, name_ = "", pos_ = [0,0,0], ori_ = [1,0,0]):
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

    def rotate(self):
        layer = []
        for y in (-1,1):
            for z in (-1,1):
                pos = (1, y, z)
                layer.append(self.posMap[pos])

        

def main():
    o = Cell()

if __name__ == "__main__":
    main()
