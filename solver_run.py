from solver import Agent
import tensorflow as tf
import numpy as np
from cube import Cube2x2

def Run():
    tf.reset_default_graph()
    save_path = './cube2x2_solve_sess'
    agent = Agent(0.0001)

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

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, save_path)

        success = 0.0
        total = 1
        encode_success = 0.0
        encode_total = 0.0
        for step in range(int(total)):
            test_cube = Cube2x2()
            num_actions = 1
            actions_index = np.random.randint(0, len(cube_action_map), num_actions)
            actions = list(np.array(list(cube_action_map.values()))[actions_index])

            print('*** Testing ***')
            print('test_cube obs: ', test_cube.GetObservation())
            for a in actions:
                print('shuffle action: ', a)
                test_cube.rotate(a)
                print('test_cube obs: ', test_cube.GetObservation())

            if test_cube.IsDone():
                success += 1
                continue

            print('NN action: ')
            for i in range(10):
                a = sess.run([agent.chosen_action, agent.output], feed_dict={agent.state_in: [test_cube.GetObservation()]})
                print('cube: ', test_cube.GetObservation())
                print('output : ', a[1])
                encode_total += 1
                print('recover action: ', cube_action_map[a[0][0]])
                test_cube.rotate(cube_action_map[a[0][0]])
                if test_cube.IsDone():
                    break
            if test_cube.IsDone():
                success += 1
        print('encode success rate: ', encode_success / encode_total)
        print('success rate: ', success / total)

if __name__ == "__main__":
    Run()
