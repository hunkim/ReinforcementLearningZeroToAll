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
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


# Setting up our environment
sess = tf.Session()
sess.run(tf.global_variables_initializer())

num_episodes = 1000
# This list will contain episode rewards from the most recent 100 games
# Clear Condition: Average reward per episode >= 195.0 over 100 games
EPISODE_100_REWARD_LIST = []
for i in range(num_episodes):

    # Clear out game variables
    xs = np.empty(shape=[0, input_size])
    ys = np.empty(shape=[0, output_size])
    rewards = np.empty(shape=[0, 1])

    reward_sum = 0
    state = env.reset()

    while True:
        # Append the observations to our batch
        x = np.reshape(state, [1, input_size])

        # Run the neural net to determine output
        action_prob = sess.run(action_pred, feed_dict={X: x})        
        action = np.random.choice(np.arange(output_size), p=action_prob[0])
    
        # Append the observations and outputs for learning
        xs = np.vstack([xs, x])
        y = np.zeros(output_size)
        y[action] = 1
        
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
            EPISODE_100_REWARD_LIST.append(reward_sum)
            if len(EPISODE_100_REWARD_LIST) > 100:
                EPISODE_100_REWARD_LIST = EPISODE_100_REWARD_LIST[1:]
            break


    # Print status
    print(f"[Episode {i:>}] Reward: {reward_sum:>4} Loss: {l:>5.5}")
    
    if np.mean(EPISODE_100_REWARD_LIST) >= 195.0:
        print(f"Game Cleared within {i} steps with the average reward: {np.mean(EPISODE_100_REWARD_LIST)}")
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

sess.close()