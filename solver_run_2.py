# from solver import Agent
# import tensorflow as tf
import numpy as np
from cube import Cube2x2
import pickle

def Run():
    cube_action_map = {
            0: 'R',
            1: 'r',
            2: 'L',
            3: 'l',
            4: 'F',
            5: 'f',
            6: 'B',
            7: 'b',
            8: 'U',
            9: 'u',
            10: 'D',
            11: 'd',
            }

    history_map_path = './history_map'
    history = pickle.load(open(history_map_path, 'rb'))

    success = 0.0
    total = 5
    encode_success = 0.0
    encode_total = 0.0
    for step in range(int(total)):
        test_cube = Cube2x2()
        num_actions = 2
        actions_index = np.random.randint(0, len(cube_action_map), num_actions)
        actions = list(np.array(list(cube_action_map.values()))[actions_index])

        print('Testing: ')
        for a in actions:
            print('shuffle action: ', a)
            test_cube.rotate(a)

        if test_cube.IsDone():
            success += 1
            continue

        print(' => Q action: ')
        for i in range(20):
            max_score = 0.0
            action = -1
            for a in range(12):
                key = (tuple(test_cube.GetObservation()), a)
                if key in history and max_score < history[key][2]:
                    max_score = history[key][2]
                    action = a
            if action < 0:
                break
            print('recover action: ', cube_action_map[action])
            test_cube.rotate(cube_action_map[action])
            print('cube after recover: ', test_cube.GetObservation())
            if test_cube.IsDone():
                print('Done')
                break
        if test_cube.IsDone():
            success += 1
    print('success rate: ', success / total)

if __name__ == "__main__":
    Run()
