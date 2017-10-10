import unittest
import solver
import tensorflow as tf
import cube
import numpy as np

class TestSolver(unittest.TestCase):
    def setUp(self):
        s_size = 24
        self.cube = cube.Cube2x2()

    def test_agent(self):
        tf.reset_default_graph()
        self.agent = solver.Agent()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            obs = self.cube.GetObservation()
            routput = tf.gather(tf.reshape(self.agent.output, [-1]),
                    self.agent.chosen_action)
            res = sess.run([self.agent.chosen_action, routput], feed_dict={self.agent.state_in:
                [obs, obs]})

            print('res: ', res)

    def test_update(self):
        tf.reset_default_graph()
        agent = solver.Agent()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            history = []
            history.append([self.cube.GetObservation(), 0, 95.0])
            history.append([self.cube.GetObservation(), 3, 100.0])
            history = np.array(history)
            print('history: ', history)
            feed_dict = {agent.reward_holder: list(history[:,2]),
                            agent.action_holder: list(history[:,1]),
                            agent.state_in: list(history[:,0])}
            print('feed_dict: ', feed_dict)
            res = sess.run(agent.update_batch, feed_dict=feed_dict)
            print(res)
