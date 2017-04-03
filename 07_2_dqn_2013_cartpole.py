"""
This code is based on:
https://github.com/hunkim/DeepRL-Agents

CF https://github.com/golbin/TensorFlow-Tutorials
"""
import numpy as np
import tensorflow as tf
import random
import dqn
from collections import deque

import gym
env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, 'gym-results/', force=True)
# Constants defining our neural network
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

# Q(s, a) = r + discount_rate * max Q(s_next, a)
DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
MAX_EPISODE = 5000
BATCH_SIZE = 32

# minimum epsilon for epsilon greedy
MIN_E = 0.01
# if epsilon decaying_episode = 100
# epsilon lineary decreases to MIN_E over 100 episodes
EPSILON_DECAYING_EPISODE = MAX_EPISODE * 0.1


def bot_play(mainDQN):
    """ Run a single episode with the DQN agent

    Parameters
    ----------
    mainDQN : DQN agent

    Returns
    ----------
    reward : float
        Episode reward is returned
    """

    state = env.reset()
    reward_sum = 0

    while True:
        env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break


def simple_replay_train(DQN, train_batch):
    """ Prepare X_batch, y_batch and train them

    Recall our loss function is
        target = reward + discount * max Q(s',a)
                 or reward if done early

        Loss function: [target - Q(s, a)]^2

    Hence,

        X_batch is a state list
        y_batch is reward + discount * max Q
                   or reward if terminated early

    Parameters
    ----------
    DQN : DQN Agent

    train_batch : list, [item_1, item_2, ..., item_batchsize]
        where item_i is also a list
        item_i = [state, action, reward, next_state, done]

    """
    # We use numpy to vectorize operations
    tmp = np.asarray(train_batch)

    # state_array.shape = (batch_size, 4)
    state_array = np.vstack(tmp[:, 0])

    # action_array.shape = (batch_size, )
    action_array = tmp[:, 1].astype(np.int32)

    # reward_array.shape = (batch_size, )
    reward_array = tmp[:, 2]

    # next_state_array.shape = (batch_size, 4)
    next_state_array = np.vstack(tmp[:, 3])

    # done_array.shape = (batch_size, )
    done_array = tmp[:, 4].astype(np.int32)

    X_batch = state_array
    y_batch = DQN.predict(state_array)

    # We use a vectorized operation
    target = reward_array + DISCOUNT_RATE * np.max(DQN.predict(next_state_array), 1) * (1 - done_array)
    y_batch[np.arange(len(X_batch)), action_array] = target

    # Train our network using target and predicted Q values on each episode
    return DQN.update(X_batch, y_batch)


def annealing_epsilon(episode, min_e, max_e, target_episode):
    """Return an linearly annealed epsilon

    Parameters
    ----------

        (epsilon)
            |
   max_e ---|\
            | \
            |  \
            |   \
   min_e ---|____\_______________(episode)
                 |
                target_episode

    slope = (min_e - max_e) / (target_episode)
    intercept = max_e

    e = slope * episode + intercept
    """

    slope = (min_e - max_e) / (target_episode)
    intercept = max_e

    return max(min_e, slope * episode + intercept)


def main():
    """
    pseudocode

    For episode = 1, ..., M

        s = initital state

        For t = 1, ..., T

            action = argmax Q(s, a)

            Get s2, r, d by playing the action

            save [s,a,r,s2,d] to memory

            take a minibatch from memory

            y = r
                or r + d * max Q(s2, ...)

            Perform train step on (y - Q(s, a))^2

    """

    # store the previous observations in replay memory
    replay_buffer = deque()

    # Check whether we clear the game
    # CartPole Clear Condition
    # Avg Reward >= 195 over 100 games
    # We will choose more strict condition by setting 199
    last_100_game_reward = deque()

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE)
        init = tf.global_variables_initializer()
        sess.run(init)

        for episode in range(MAX_EPISODE):
            e = annealing_epsilon(episode, MIN_E, 1.0, EPSILON_DECAYING_EPISODE)
            done = False
            state = env.reset()

            step_count = 0
            while not done:

                if np.random.rand() < e:
                    action = env.action_space.sample()

                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)

                if done:
                    reward = -1

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1

                if step_count % 4 == 0 and len(replay_buffer) > BATCH_SIZE:
                    # Minibatch works better
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    simple_replay_train(mainDQN, minibatch)

            print("[Episode {:>5}]  steps: {:>5} e: {:>5.2f}".format(episode, step_count, e))

            last_100_game_reward.append(step_count)

            if len(last_100_game_reward) > 100:

                last_100_game_reward.popleft()

                avg_reward = np.mean(last_100_game_reward)
                if avg_reward > 199.0:
                    print("Game Cleared within {} episodes with avg reward {}".format(episode, avg_reward))
                    break

        # Test run 5 times
        for _ in range(5):
            bot_play(mainDQN)


if __name__ == "__main__":
    main()
