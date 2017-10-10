import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
from cube import Cube2x2

def DiscountRewards(r):
    gamma = 0.2
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Agent():
    def __init__(self, learning_rate=0.01, learning_rate_encode=0.001):
        s_size = 24
        h_size = 720
        a_size = 12
        state_depth = 6
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.uint8)
        self.state = tf.one_hot(self.state_in, depth=state_depth)
        self.state = tf.reshape(self.state, [tf.shape(self.state_in)[0], s_size*state_depth])

        hidden = slim.fully_connected(self.state, h_size, biases_initializer=None, activation_fn=tf.nn.relu, scope='hidden1')
        for i in range(2):
            hidden = slim.fully_connected(hidden, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, a_size,activation_fn=tf.nn.softmax,biases_initializer=None, scope='output')
        self.chosen_action = tf.argmax(self.output, 1)

        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.chosen_outputs = tf.gather(tf.reshape(self.output, [-1]), self.chosen_action)
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.action_holder)

        tvars = tf.trainable_variables()
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder) + tf.reduce_mean((self.chosen_outputs - self.responsible_outputs) * self.reward_holder)
        self.gradients = tf.gradients(self.loss, tvars)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_batch = optimizer.apply_gradients(zip(self.gradients, tvars))

def Train():
    tf.reset_default_graph()
    total_episodes = 1000
    max_ep = 25

    cube = Cube2x2()
    agent = Agent(0.00001)

    save_path = './cube2x2_solve_sess'
    history_map_path = './history_map'
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        history_map = {}

        load = False
        load_history_map = True
        if load:
            saver.restore(sess, save_path)
        if load_history_map:
            history_map = pickle.load(open(history_map_path, 'rb'))

        i = 0
        train_history = []
        encoder_ready = False
        while True:
            running_reward = 0
            pre_num_actions = 2
            if i == 0 and not load_history_map:
                mmax = 1000
            else:
                mmax = 5
            for m in range(mmax):
                history = []
                cube = Cube2x2()
                for j in range(max_ep):
                    obs = cube.GetObservation()
                    action = np.random.randint(0, cube.GetNumOfActions(), 1)[0]
 
                    cube.rotate(cube.GetActionById(action))
                    done = cube.IsDone()

                    if done:
                        reward = 100.0
                    else:
                        reward = 0.0
                    history.append([obs, action, reward])

                    if done:
                        break
                if done:
                    continue

                if not done:
                    new_history = []
                    for item in reversed(history):
                        obs = cube.GetObservation()
                        action = cube.GetActionById(item[1])
                        action = cube.GetReverseAction(action)
                        cube.rotate(action)
                        if cube.IsDone():
                            reward = 1.0
                        else:
                            reward = 0.0
                        new_history.append([obs, cube.GetIdForAction(action), reward])
                    history = new_history

                    if not cube.IsDone():
                        print(history)
                        print(cube.GetObservation())
                        raise 'Error not done'

                history = np.array(history)
                history[:,2] = DiscountRewards(history[:,2])

                if len(train_history) == 0:
                    train_history = history
                    for item in train_history:
                        history_map[(tuple(item[0]), item[1])] = item
                else:
                    for item in history:
                        key = (tuple(item[0]), item[1])
                        if key not in history_map:
                            history_map[key] = item
                        else:
                            record = item
                            record[2] = max(item[2], history_map[key][2])
                            history_map[key] = record

            pickle.dump(history_map, open(history_map_path, 'wb'))
            
            train_batch_size = min(3000000, 10 * np.log(len(history_map.values())))
            if (train_batch_size < 1):
                continue

            print('*** batch updating ***')
            update_batch_num = 1
            for iter in range(update_batch_num):
                train_batch_index = np.random.choice(len(history_map.values()), train_batch_size)
                train_batch = np.array(list(history_map.values()))[train_batch_index]

                for iter2 in range(1):
                    feed_dict = {agent.reward_holder: list(train_batch[:,2]),
                            agent.action_holder: list(train_batch[:,1]),
                            agent.state_in: list(train_batch[:,0])}

                    _ = sess.run(agent.update_batch, feed_dict=feed_dict)
                if iter % 10 == 0:
                    train_batch = np.array(list(history_map.values()))
                    feed_dict = {agent.reward_holder: list(train_batch[:,2]),
                            agent.action_holder: list(train_batch[:,1]),
                            agent.state_in: list(train_batch[:,0])}

                    loss = sess.run(agent.loss, feed_dict=feed_dict)
                    print('loss: ', loss)


            if i % 5 == 0:
                print('Iter: ', i, ' training history size: ', len(history_map.values()))
                saver.save(sess, save_path)
            i += 1

        saver.save(sess, save_path)

def main():
    Train()

if __name__ == "__main__":
    main()
