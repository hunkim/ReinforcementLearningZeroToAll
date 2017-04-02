import tensorflow as tf
import numpy as np
from dqn import DQN


class TestDQN:
    def setup_method(self, method):
        print("Session open")
        tf.reset_default_graph()
        self.sess = tf.Session()

    def teardown_method(self, method):
        print("Sesson close")
        self.sess.close()

    def test_one_agent(self):
        agent = DQN(self.sess, 4, 2)
        assert isinstance(agent, DQN) is True

        assert hasattr(agent, "_X")
        assert hasattr(agent, "_Y")
        assert hasattr(agent, "_loss")
        assert hasattr(agent, "_train")

    def run_init(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def test_agent_can_take_observation(self):
        obs = np.zeros([1, 4])
        agent = DQN(self.sess, 4, 2)
        self.run_init()
        output = agent.predict(obs)
        np.testing.assert_almost_equal(output, [[0, 0]])

        obs = np.zeros([4, ])
        output = agent.predict(obs)
        np.testing.assert_almost_equal(output, [[0, 0]])

        obs = np.zeros([32, 4])
        output = agent.predict(obs)
        np.testing.assert_almost_equal(output, [[0, 0] for _ in range(32)])

    def test_agent_can_run_update(self):
        x_stack = np.zeros([32, 4])
        y_stack = np.zeros([32, 2])

        agent = DQN(self.sess, 4, 2)
        self.run_init()

        output = agent.update(x_stack, y_stack)
        assert output[0] == 0

        x_stack = np.zeros([1, 4])
        y_stack = np.zeros([1, 2])

        output = agent.update(x_stack, y_stack)
        assert output[0] == 0
