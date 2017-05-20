"""
Yet Another Pong

Vanilla Policy Gradient implementation

(1) Pong's state is (210, 160, 3)
(2) After `pipeline(image)`, it becomes (80, 80, 1)
(3) The model uses an input of `state_diff` = `new_state` - `old_state`
(4) It assumes there exists 2 actions.

        Pong's original action space is the following:
            0, 1 : do nothing
            2, 4 : move up
            3, 5 : move down

        In this file, it uses {2: move up, 3: move down} only

        It gets rid of unnecessary complexity.
"""
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from functools import partial
from scipy.misc import imresize

import os


def plot_image(image):
    """Plot an image

    If an image is a grayscale image,
    plot in `gray` cmap.
    Otherwise, regular RGB plot.

    Args:
        image (2-D or 3-D array): (H, W) or (H, W, C)
    """
    image = np.squeeze(image)
    shape = image.shape

    if len(shape) == 2:
        plt.imshow(image, cmap="gray")

    else:
        plt.imshow(image)

    plt.show()


def pipeline(image, new_HW, height_range=(35, 193), bg=(144, 72, 17)):
    """Returns a preprocessed image

    (1) Crop image (top and bottom)
    (2) Remove background & grayscale
    (3) Reszie to smaller image

    Args:
        image (3-D array): (H, W, C)
        new_HW (tuple): New image size (height, width)
        height_range (tuple): Height range (H_begin, H_end) else cropped
        bg (tuple): Background RGB Color (R, G, B)

    Returns:
        image (3-D array): (H, W, 1)
    """
    image = crop_image(image, height_range)
    image = resize_image(image, new_HW)
    image = kill_background_grayscale(image, bg)

    image = np.expand_dims(image, axis=2)
    return image


def resize_image(image, new_HW):
    """Returns a resized image

    Args:
        image (3-D array): Numpy array (H, W, C)
        new_HW (tuple): Target size (height, width)

    Returns:
        image (3-D array): Resized image (height, width, C)
    """
    return imresize(image, new_HW, interp="nearest")


def crop_image(image, height_range=(35, 195)):
    """Crops top and bottom

    Args:
        image (3-D array): Numpy image (H, W, C)
        height_range (tuple): Height range between (min_height, max_height)
            will be kept

    Returns:
        image (3-D array): Numpy image (max_H - min_H, W, C)
    """
    h_beg, h_end = height_range
    return image[h_beg:h_end, ...]


def kill_background_grayscale(image, bg):
    """Make the background 0

    Args:
        image (3-D array): Numpy array (H, W, C)
        bg (tuple): RGB code of background (R, G, B)

    Returns:
        image (2-D array): Binarized image of shape (H, W)
            The background is 0 and everything else is 1
    """
    H, W, _ = image.shape

    R = image[..., 0]
    G = image[..., 1]
    B = image[..., 2]

    cond = (R == bg[0]) & (G == bg[1]) & (B == bg[2])

    image = np.zeros((H, W))
    image[~cond] = 1

    return image


class Agent(object):

    def __init__(self, input_dim, output_dim, logdir="logdir", checkpoint_dir="checkpoints"):
        """Agent class

        Args:
            input_dim (tuple): The input shape (H, W, C)
            output_dim (int): Number of actions
            logdir (str): Directory to save `summary`
            checkpoint_dir (str): Directory to save `model.ckpt`

        Notes:

            It has two methods.

                `choose_action(state)`
                    Will return an action given the state

                `train(state, action, reward)`
                    Will train on given `states`, `actions`, `rewards`

            Private methods has two underscore prefixes
        """
        self.input_dim = list(input_dim)
        self.output_dim = output_dim
        self.gamma = 0.99
        self.entropy_coefficient = 0.01
        self.RMSPropdecay = 0.99
        self.learning_rate = 0.001

        self.checkpoint_dir = checkpoint_dir
        self.__build_network(self.input_dim, self.output_dim)

        if logdir is not None:
            self.__build_summary_op(logdir)
        else:
            self.summary_op = None

        if checkpoint_dir is not None:
            self.saver = tf.train.Saver()

            maybe_path = os.path.join(self.checkpoint_dir, "model.ckpt")
            if os.path.exists(self.checkpoint_dir) and tf.train.checkpoint_exists(maybe_path):
                print("Restored {}".format(maybe_path))
                sess = tf.get_default_session()
                self.saver.restore(sess, maybe_path)

            else:
                print("No model is found")
                os.makedirs(checkpoint_dir, exist_ok=True)

    def __build_network(self, input_dim, output_dim):

        self.global_step = tf.train.get_or_create_global_step()

        self.X = tf.placeholder(tf.float32, shape=[None, *input_dim], name='state')
        self.action = tf.placeholder(tf.uint8, shape=[None], name="action")
        action_onehot = tf.one_hot(self.action, output_dim, name="action_onehot")
        self.reward = tf.placeholder(tf.float32, shape=[None], name="reward")

        net = self.X

        with tf.variable_scope("layer1"):
            net = tf.layers.conv2d(net,
                                   filters=16,
                                   kernel_size=(8, 8),
                                   strides=(4, 4),
                                   name="conv")
            net = tf.nn.relu(net, name="relu")

        with tf.variable_scope("layer2"):
            net = tf.layers.conv2d(net,
                                   filters=32,
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   name="conv")
            net = tf.nn.relu(net, name="relu")

        with tf.variable_scope("fc1"):
            net = tf.contrib.layers.flatten(net)
            net = tf.layers.dense(net, 256, name='dense')
            net = tf.nn.relu(net, name='relu')

        with tf.variable_scope("fc2"):
            net = tf.layers.dense(net, output_dim, name='dense')

        self.action_prob = tf.nn.softmax(net, name="action_prob")

        log_action_prob = tf.reduce_sum(self.action_prob * action_onehot, axis=1)
        log_action_prob = tf.log(log_action_prob + 1e-7)

        entropy = - self.action_prob * tf.log(self.action_prob + 1e-7)
        self.entropy = tf.reduce_sum(entropy, axis=1)

        loss = -log_action_prob * self.reward - self.entropy * self.entropy_coefficient
        self.loss = tf.reduce_mean(loss)

        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                              decay=self.RMSPropdecay)

        self.train_op = optimizer.minimize(loss,
                                           global_step=self.global_step)

    def __build_summary_op(self, logdir):
        tf.summary.histogram("Action Probability Histogram", self.action_prob)
        tf.summary.histogram("Entropy", self.entropy)
        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("Mean Reward", tf.reduce_mean(self.reward))

        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    def choose_action(self, S):
        shape = S.shape

        if len(shape) == 3:
            S = np.expand_dims(S, axis=0)

        np.testing.assert_equal(S.shape[1:], self.input_dim)

        sess = tf.get_default_session()
        action_prob = sess.run(self.action_prob,
                               feed_dict={self.X: S})
        action_prob = np.squeeze(action_prob)
        return np.random.choice(np.arange(self.output_dim) + 2, p=action_prob)

    def train(self, S, A, R):
        S = np.array(S)
        A = np.array(A)
        R = np.array(R)
        np.testing.assert_equal(S.shape[1:], self.input_dim)
        assert len(A.shape) == 1, "A.shape = {}".format(A.shape)
        assert len(R.shape) == 1, "R.shape = {}".format(R.shape)

        R = discount_reward(R, gamma=self.gamma)
        R -= np.mean(R)
        R /= np.std(R) + 1e-7

        A = A - 2

        sess = tf.get_default_session()

        _, summary_op, global_step_value = sess.run([self.train_op,
                                                     self.summary_op,
                                                     self.global_step],
                                                    feed_dict={self.X: S,
                                                               self.action: A,
                                                               self.reward: R})

        if self.summary_op is not None:
            self.summary_writer.add_summary(summary_op, global_step_value)

    def save(self):
        sess = tf.get_default_session()
        path = os.path.join(self.checkpoint_dir, "model.ckpt")
        self.saver.save(sess, path)


def discount_reward(rewards, gamma=0.99):
    """Returns discounted rewards

    Args:
        rewards (1-D array): Reward array
        gamma (float): Discounted rate

    Returns:
        discounted_rewards: same shape as `rewards`

    Notes:
        In Pong, when the reward can be {-1, 0, 1}.

        However, when the reward is either -1 or 1,
        it means the game has been reset.

        Therefore, it's necessaray to reset `running_add` to 0
        whenever the reward is nonzero
    """
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        if rewards[t] != 0:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add

    return discounted_r


def run_episode(env, agent, pipeline):
    """Runs one episode and returns a total reward

    Args:
        env (gym.env): Gym Environment
        agent (Agent): Agent Player
        pipeline (function): Preprocessing function.
            processed_image = pipeline(image)

    Returns:
        total_reward (int): Total reward earned in an episode.
    """
    states = []
    actions = []
    rewards = []

    old_s = env.reset()
    old_s = pipeline(old_s)

    done = False
    total_reward = 0
    step_counter = 0

    state_diff = old_s

    while not done:

        action = agent.choose_action(state_diff)
        new_s, r, done, info = env.step(action)
        total_reward += r

        states.append(state_diff)
        actions.append(action)
        rewards.append(r)

        new_s = pipeline(new_s)
        state_diff = new_s - old_s
        old_s = new_s

        if r == -1 or r == 1 or done:
            step_counter += 1

            if step_counter > 10 or done:
                step_counter = 0
                # Agent expects numpy array
                agent.train(states, actions, rewards)

                states, actions, rewards = [], [], []

    return total_reward


def main():
    try:
        env = gym.make("Pong-v0")
        env = gym.wrappers.Monitor(env, "monitor", force=True)
        action_dim = 2

        tf.reset_default_graph()
        sess = tf.InteractiveSession()

        new_HW = [80, 80]
        repeat = 1
        pipeline_fn = partial(pipeline, new_HW=new_HW, height_range=(35, 195), bg=(144, 72, 17))

        agent = Agent(new_HW + [repeat],
                      output_dim=action_dim,
                      logdir='logdir/train',
                      checkpoint_dir="checkpoints")

        init = tf.global_variables_initializer()
        sess.run(init)

        episode = 1

        while True:
            episode_reward = run_episode(env, agent, pipeline_fn)
            print(episode, episode_reward)

            episode += 1

    finally:
        agent.save()

        env.close()
        sess.close()


def debug_mode():
    pipeline_fn = partial(pipeline, new_HW=(50, 50), height_range=(35, 195), bg=(144, 72, 17))
    try:

        env = gym.make("Pong-v0")
        env.reset()

        for _ in range(50):

            s = env.step(env.action_space.sample())[0]

            plot_image(np.squeeze(pipeline_fn(s)))

    finally:

        env.close()


if __name__ == '__main__':
    main()
    # debug_mode()
