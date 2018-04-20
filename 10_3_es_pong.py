"""
    Evolution Strategies to play Pong first round,
    to play whole game remove "if reward != 0: done = True" from f(N, j) function,
    but it will slow down each update by 21 times,
    first three cnn layers same as in dqn paper 2015,
    will see prevalent winning at iter ~150, with continuing improvement
"""

import numpy as np
import tensorflow as tf
import gym
import sys
import os

env = gym.make("Pong-v0")
tf.set_random_seed(10)
np.random.seed(10)

input_size = 80 * 80 * 4
action_space = env.action_space.n
CHECK_POINT_DIR = './tensorboard/pong-aver'
aver_reward = None
aver_mean_pop = None
play_the_game = False

npop = 50
sigma = 0.1
alpha = 3*1e-2

print npop, sigma, alpha

shapes = {'W_conv1': [8, 8, 4, 32], 'b_conv1': [32], 'W_conv2': [4, 4, 32, 64], 'b_conv2': [64],
          'W_conv3': [3, 3, 64, 64], 'b_conv3': [64], 'W_fc1': [10 * 10 * 64, action_space], 'b_fc1': [action_space]}

model = {}
model['W_conv1'] = tf.get_variable("W_conv1", shape=[8, 8, 4, 32], initializer=tf.contrib.layers.xavier_initializer())
model['b_conv1'] = tf.Variable(tf.zeros([32]), name="b_conv1")
model['W_conv2'] = tf.get_variable("W_conv2", shape=[4, 4, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
model['b_conv2'] = tf.Variable(tf.zeros([64]), name="b_conv2")
model['W_conv3'] = tf.get_variable("W_conv3", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
model['b_conv3'] = tf.Variable(tf.zeros([64]), name="b_conv3")
model['W_fc1'] = tf.get_variable("W_fc1", shape=[10 * 10 * 64, action_space],
                        initializer=tf.contrib.layers.xavier_initializer())
model['b_fc1'] = tf.Variable(tf.zeros([action_space]), name='b_fc1')

X = tf.placeholder(tf.float32, [None, input_size], name="input_x")
x_image = tf.reshape(X, [-1, 80, 80, 4])
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, model['W_conv1'], strides=[1, 4, 4, 1], padding='SAME') + model['b_conv1'])
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, model['W_conv2'], strides=[1, 2, 2, 1], padding='SAME') + model['b_conv2'])
h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, model['W_conv3'], strides=[1, 1, 1, 1], padding='SAME') + model['b_conv3'])
h_conv3_flat = tf.reshape(h_conv3, [-1, 10 * 10 * 64], name="h_pool2_flat")
action_pred = tf.nn.softmax(tf.matmul(h_conv3_flat, model['W_fc1']) + model['b_fc1'])

N_placeholder = {k: tf.placeholder(tf.float32, shape=model[k].get_shape()) for k in model}

model_try = {k: tf.add(model[k], N_placeholder[k]) for k in model}

X_try = tf.placeholder(tf.float32, [None, input_size], name="input_x")
x_image_try = tf.reshape(X_try, [-1, 80, 80, 4])
h_conv1_try = tf.nn.relu(tf.nn.conv2d(x_image_try, model_try['W_conv1'], strides=[1, 4, 4, 1], padding='SAME') + model_try['b_conv1'])
h_conv2_try = tf.nn.relu(tf.nn.conv2d(h_conv1_try, model_try['W_conv2'], strides=[1, 2, 2, 1], padding='SAME') + model_try['b_conv2'])
h_conv3_try = tf.nn.relu(tf.nn.conv2d(h_conv2_try, model_try['W_conv3'], strides=[1, 1, 1, 1], padding='SAME') + model_try['b_conv3'])
h_conv3_flat_try = tf.reshape(h_conv3_try, [-1, 10 * 10 * 64], name="h_pool2_flat")
action_pred_try = tf.nn.softmax(tf.matmul(h_conv3_flat_try, model_try['W_fc1']) + model_try['b_fc1'])

def prepro(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()

def get_action(state, N, j, original):
    if original:
        action_prob = sess.run(action_pred, feed_dict={X: state})
        action_prob = np.squeeze(action_prob)
        action = np.random.choice(action_space, size=1, p=action_prob)[0]
        return action

    feed_dict = {X_try: state}
    for k in model:
        feed_dict[N_placeholder[k]] = sigma * N[k][j]

    action_prob = sess.run(action_pred_try, feed_dict=feed_dict)
    action_prob = np.squeeze(action_prob)
    action = np.random.choice(action_space, size=1, p=action_prob)[0]
    return action

def f(N, j, original=False, render=False):
    state = env.reset()
    state = prepro(state)
    s_t = np.array([state, state, state, state])
    total_reward = 0

    while True:
        if render: env.render()

        x = np.reshape(s_t, [1, input_size])
        action = get_action(x, N, j, original)
        state, reward, done, info = env.step(action)

        state = prepro(state)
        s_t = np.array([state, s_t[0], s_t[1], s_t[2]])
        total_reward += reward

        if reward != 0: done = True
        if done:
            break
    return total_reward

assign_placeholder = {k: tf.placeholder(tf.float32, shape=model[k].get_shape()) for k in model}
update_op2 = {k: tf.assign(model[k],
                           tf.add(model[k], assign_placeholder[k]))
              for k in model}

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECK_POINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
    try:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    except:
        print("Error on loading old network weights")
else:
    print("Could not find old network weights")

if play_the_game:
    for i_episode in xrange(100):
        print f({}, -1, original=True, render=True)
    sys.exit('demo finished')

for i_episode in xrange(100001):
    R = np.zeros(npop)
    N = {}
    for k in model:
        N[k] = np.random.randn(npop, *shapes[k])

    for j in range(npop):
        R[j] = f(N, j)
    A = (R - np.mean(R)) / (np.std(R) + 1e-5)

    for k in model:
        transpose_order = range(1, len(N[k].shape))
        transpose_order.append(0)
        update_tensor = alpha/(npop*sigma) * np.dot(N[k].transpose(transpose_order), A)

        sess.run(update_op2[k], feed_dict={assign_placeholder[k]: update_tensor})

    cur_reward = f({}, -1, original=True)
    aver_reward = 0.9*aver_reward + 0.1*cur_reward if aver_reward is not None else cur_reward
    mean_pop = np.mean(R)
    aver_mean_pop = 0.9*aver_mean_pop + 0.1*mean_pop if aver_mean_pop is not None else mean_pop

    print('iter %d, mean_pop %.2f, aver_mean_pop %.2f, cur_reward %.2f, aver_reward %.2f' %
          (i_episode, mean_pop, aver_mean_pop, cur_reward, aver_reward))

    if i_episode % 10 == 0:
        print("Saving network...")
        if not os.path.exists(CHECK_POINT_DIR):
            os.makedirs(CHECK_POINT_DIR)
        saver.save(sess, CHECK_POINT_DIR + "/pong", global_step=i_episode)
    if i_episode == 50: alpha = 1e-2

"""
iter 0, mean_pop -1.00, aver_mean_pop -1.00, cur_reward -1.00, aver_reward -1.00
Saving network...
iter 1, mean_pop -1.00, aver_mean_pop -1.00, cur_reward -1.00, aver_reward -1.00
iter 2, mean_pop -0.96, aver_mean_pop -1.00, cur_reward -1.00, aver_reward -1.00
...
iter 254, mean_pop 0.64, aver_mean_pop 0.64, cur_reward 1.00, aver_reward 1.00
iter 255, mean_pop 0.68, aver_mean_pop 0.65, cur_reward 1.00, aver_reward 1.00
iter 256, mean_pop 0.76, aver_mean_pop 0.66, cur_reward 1.00, aver_reward 1.00
"""