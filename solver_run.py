from solver import Agent
import tensorflow as tf
import numpy as np
from cube import Cube2x2

def Run():
    tf.reset_default_graph()
    save_path = './cube2x2_solve_sess'
    agent = Agent(0.0001)

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
            cube = Cube2x2()
            num_actions = 1
            actions_index = np.random.randint(0, cube.GetNumOfActions(), num_actions)
            actions = list(np.array(cube.GetActions())[actions_index])

            print('*** Testing ***')
            print('cube obs: ', cube.GetObservation())
            for a in actions:
                print('shuffle action: ', a)
                cube.rotate(a)
                print('cube obs: ', cube.GetObservation())

            if cube.IsDone():
                success += 1
                continue

            print('NN action: ')
            for i in range(10):
                a = sess.run([agent.chosen_action, agent.output], feed_dict={agent.state_in: [cube.GetObservation()]})
                print('cube: ', cube.GetObservation())
                print('output : ', a[1])
                encode_total += 1
                action = cube.GetActionById(a[0][0])
                print('recover action: ', action)
                cube.rotate(action)
                if cube.IsDone():
                    break
            if cube.IsDone():
                success += 1
        print('encode success rate: ', encode_success / encode_total)
        print('success rate: ', success / total)

if __name__ == "__main__":
    Run()
