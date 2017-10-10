# from solver import Agent
# import tensorflow as tf
import numpy as np
from cube import Cube2x2
import pickle

def Run():
    history_map_path = './history_map'
    history = pickle.load(open(history_map_path, 'rb'))

    success = 0.0
    total = 5
    encode_success = 0.0
    encode_total = 0.0
    for step in range(int(total)):
        cube = Cube2x2()
        num_actions = 2
        actions_index = np.random.randint(0, cube.GetNumOfActions(), num_actions)
        actions = list(np.array(cube.GetActions())[actions_index])

        print('Testing: ')
        for a in actions:
            print('shuffle action: ', a)
            cube.rotate(a)

        if cube.IsDone():
            success += 1
            continue

        print(' => Q action: ')
        for i in range(20):
            max_score = 0.0
            action = -1
            for a in range(12):
                key = (tuple(cube.GetObservation()), a)
                if key in history and max_score < history[key][2]:
                    max_score = history[key][2]
                    action = a
            if action < 0:
                break
            action = cube.GetActionById(action)
            print('recover action: ', action)
            cube.rotate(action)
            print('cube after recover: ', cube.GetObservation())
            if cube.IsDone():
                print('Done')
                break
        if cube.IsDone():
            success += 1
    print('success rate: ', success / total)

if __name__ == "__main__":
    Run()
