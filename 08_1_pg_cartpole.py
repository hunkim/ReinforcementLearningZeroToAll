'''
This code is based on:
https://github.com/hunkim/DeepRL-Agents
http://karpathy.github.io/2016/05/31/rl/
'''
import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')

hidden_layer_neurons = 24
learning_rate = 1e-2

# Constants defining our neural network
input_size = env.observation_space.shape[0]
output_size = 1  # logistic regression, one p output

X = tf.placeholder(tf.float32, [None, input_size], name="input_x")

# First layer of weights
W1 = tf.get_variable("W1", shape=[input_size, hidden_layer_neurons],
                     initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(X, W1))

# Second layer of weights
W2 = tf.get_variable("W2", shape=[hidden_layer_neurons, output_size],
                     initializer=tf.contrib.layers.xavier_initializer())
action_pred = tf.nn.sigmoid(tf.matmul(layer1, W2))

# Y (fake) and advantages (rewards)
Y = tf.placeholder(tf.float32, [None, output_size], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

# Loss function: log_likelihood * advantages
#log_lik = -tf.log(Y * action_pred + (1 - Y) * (1 - action_pred))     # using author(awjuliani)'s original cost function (maybe log_likelihood)
log_lik = -Y*tf.log(action_pred) - (1 - Y)*tf.log(1 - action_pred)    # using logistic regression cost function
loss = tf.reduce_sum(log_lik * advantages)

# Learning
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r

# Testing Code
# It's always recommended to test your code
input = [1, 1, 1]
output = discount_rewards(input)
expect = [1 + 0.99 + 0.99**2, 1 + 0.99, 1]
np.testing.assert_almost_equal(output, expect)



# Setting up our environment
sess = tf.Session()
sess.run(tf.global_variables_initializer())

max_num_episodes = 500

# This list will contain episode rewards from the most recent 100 games
# Clear Condition: Average reward per episode >= 195.0 over 100 games
EPISODE_100_REWARD_LIST = []
for step in range(max_num_episodes):
    # Initialize x stack, y stack, and rewards
    xs = np.empty(shape=[0, input_size])
    ys = np.empty(shape=[0, 1])
    rewards = np.empty(shape=[0, 1])

    reward_sum = 0
    observation = env.reset()

    while True:
        x = np.reshape(observation, [1, input_size])

        # Run the neural net to determine output
        action_prob = sess.run(action_pred, feed_dict={X: x})

        # Determine the output based on our net, allowing for some randomness
        action = 0 if action_prob < np.random.uniform() else 1

        # Append the observations and outputs for learning
        xs = np.vstack([xs, x])
        ys = np.vstack([ys, action])  # Fake action

        # Determine the outcome of our action
        observation, reward, done, _ = env.step(action)
        rewards = np.vstack([rewards, reward])
        reward_sum += reward

        if done:
            # Determine standardized rewards
            discounted_rewards = discount_rewards(rewards)
            l, _ = sess.run([loss, train],
                            feed_dict={X: xs, Y: ys, advantages: discounted_rewards})

            EPISODE_100_REWARD_LIST.append(reward_sum)
            if len(EPISODE_100_REWARD_LIST) > 100:
                EPISODE_100_REWARD_LIST = EPISODE_100_REWARD_LIST[1:]
            break

    # Print status
    print(f"[Episode {step:>5d}] Reward: {reward_sum:>4} Loss: {l:>10.5f}")
    
    if np.mean(EPISODE_100_REWARD_LIST) >= 195:
        print(f"Game Cleared within {step} steps with the average reward: {np.mean(EPISODE_100_REWARD_LIST)}")
        break

# See our trained bot in action
observation = env.reset()
reward_sum = 0

while True:
    env.render()
    x = np.reshape(observation, [1, input_size])
    action_prob = sess.run(action_pred, feed_dict={X: x})
    action = 0 if action_prob < 0.5 else 1  # No randomness
    observation, reward, done, _ = env.step(action)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break

sess.close()