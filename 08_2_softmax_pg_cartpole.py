'''
This code is based on:
https://github.com/hunkim/DeepRL-Agents
http://karpathy.github.io/2016/05/31/rl/
'''
import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')
env._max_episode_steps = 10001

hidden_layer_neurons = 24
learning_rate = 1e-2
gamma = .99

# Constants defining our neural network
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

X = tf.placeholder(tf.float32, [None, input_size], name="input_x")

# First layer of weights
W1 = tf.get_variable("W1", shape=[input_size, hidden_layer_neurons],
                     initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(X, W1))

# Second layer of weights
W2 = tf.get_variable("W2", shape=[hidden_layer_neurons, output_size],
                     initializer=tf.contrib.layers.xavier_initializer())
action_pred = tf.nn.softmax(tf.matmul(layer1, W2))

# We need to define the parts of the network needed for learning a policy
Y = tf.placeholder(tf.float32, [None, output_size], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

print(Y, action_pred)
# Loss function, ∑ Ai*logp(yi∣xi), but we need fake lable Y due to autodiff
log_lik = -Y * tf.log(action_pred)
log_lik_adv = log_lik * advantages
loss = tf.reduce_mean(tf.reduce_sum(log_lik_adv, axis=1))

# Learning
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


def discount_rewards(r, gamma=0.99):
    """Takes 1d float array of rewards and computes discounted reward
    e.g. f([1, 1, 1], 0.99) -> [1, 0.99, 0.9801] -> [1.22 -0.004 -1.22]
    """
    d_rewards = np.array([val * (gamma ** i) for i, val in enumerate(r)])

    # Normalize/standardize rewards
    d_rewards -= d_rewards.mean()
    d_rewards /= d_rewards.std()
    return d_rewards


# Setting up our environment
sess = tf.Session()
sess.run(tf.global_variables_initializer())

num_episodes = 5000
for i in range(num_episodes):

    # Clear out game variables
    xs = np.empty(0).reshape(0, input_size)
    ys = np.empty(0).reshape(0, output_size)
    rewards = np.empty(0).reshape(0, 1)

    reward_sum = 0
    state = env.reset()

    while True:
        # Append the observations to our batch
        x = np.reshape(state, [1, input_size])

        # Run the neural net to determine output
        action_prob = sess.run(action_pred, feed_dict={X: x})
        action_prob = np.squeeze(action_prob)  # shape (?, 2) -> 2
        random_noise = np.random.uniform(0, 1, output_size)
        action = np.argmax(action_prob + random_noise)

        # Append the observations and outputs for learning
        xs = np.vstack([xs, x])
        y = np.eye(output_size)[action:action + 1]  # One hot encoding
        ys = np.vstack([ys, y])

        # Determine the outcome of our action
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        rewards = np.vstack([rewards, reward])

        if done:
            # Determine standardized rewards
            discounted_rewards = discount_rewards(rewards, gamma)
            ll, la, l, _ = sess.run([log_lik, log_lik_adv, loss, train], feed_dict={X: xs,
                                                                                    Y: ys,
                                                                                    advantages: discounted_rewards})
            # print values for debugging
            # print(1, ll, la)

            break

        if reward_sum > 10000:
            print("Solved in {} episodes!".format(i))
            break

    # Print status
    print("Average reward for episode {}: {}. Loss: {}".format(
        i, reward_sum, l))

    if reward_sum > 10000:
        break


state = env.reset()
reward_sum = 0

while True:
    env.render()

    x = np.reshape(state, [1, input_size])
    action_prob = sess.run(action_pred, feed_dict={X: x})
    action = np.argmax(action_prob)
    state, reward, done, _ = env.step(action)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break
