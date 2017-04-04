"""
Cross Entropy Method

Cross Entropy Method is a simple and efficient method
for solving a variety of estimation and optimization problems.

Psuedocode

initialize mu, sd
while not done:
    collect N samples of theta ~ N(mu, diag(sd))
    perform one episode with each theta
    select top performing samples, called elite set
    obtain a new mu and sd
end

"""
import numpy as np
import gym


env = gym.make("CartPole-v0")

INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n


def get_W_b(theta):
    """Get W and b

    Parameters
    ----------
    theta : 1-d array
        Flatten theta

    Returns
    ----------
    W : 2-d array
    b : 1-d array

    Examples
    ----------
    >>> theta = np.random.randn(5)
    >>> W, b = get_W_b(theta)
    """
    idx = INPUT_SIZE * OUTPUT_SIZE
    W = theta[:idx].reshape(INPUT_SIZE, OUTPUT_SIZE)
    b = theta[idx:].reshape(OUTPUT_SIZE)

    return W, b


def choose_action(s, W, b):
    """Return an action (argmax)

    Parameters
    ----------
    s : ndarray
        Observation (input_dim, )

    W : ndarray, (input_dim, number_of_actions)
    b : ndarray, (number_of_actions)

    Returns
    ----------
    action: int
        action index

    Examples
    ----------
    >>> s = env.reset()
    >>> W, b = get_W_b(theta)
    >>> action = choose_action(s, W, b)
    """

    action = np.dot(s, W) + b
    return np.argmax(action)


def run_episode(env, theta, render=False):
    """ Run a single episode with theta

    Parameters
    ----------
    env : gym environment
    theta : 1-d array
    render : bool, optional

    Returns
    ----------
    reward : float
        Episode reward

    Examples
    ----------
    >>> env = gym.make('CartPole-v0')
    >>> reward = run_episode(env, theta)
    """
    W, b = get_W_b(theta)
    s = env.reset()
    done = False

    reward = 0

    while not done:
        if render:
            env.render()

        a = choose_action(s, W, b)
        s2, r, done, info = env.step(a)
        reward += r
        s = s2

    return reward


def make_theta(theta_mean, theta_sd):
    """ Make a theta parameters with mean and sd

    Parameters
    ----------
    theta_mean : ndarray
        A n-d array of means

    theta_sd : nd array
        A n-d array of standard deviations

    Returns
    ----------
    theta : n-d array
        Shape (n, )

    Examples
    ----------
    >>> DIM = INPUT_SIZE * OUTPUT_SIZE + OUTPUT_SIZE
    >>> mu = np.zeros(DIM)
    >>> sd = np.ones(SD)
    >>> theta = make_theta(mu, sd)

    """
    return np.random.multivariate_normal(mean=theta_mean, cov=np.diag(theta_sd),)


def main():
    """ Every magic happens here """
    global env, INPUT_SIZE, OUTPUT_SIZE

    # Number of samples
    N = 32
    # Size of theta
    DIM = INPUT_SIZE * OUTPUT_SIZE + OUTPUT_SIZE

    # Initialize parameters
    theta_mean = np.zeros(DIM)
    theta_sd = np.ones(DIM)

    # Loop until clear the game
    #   make population with mean & sd
    #   choose elite groups
    #   obtain new mean & sd
    for _ in range(100):
        population = [make_theta(theta_mean, theta_sd) for _ in range(N)]
        reward = [run_episode(env, p) for p in population]

        sorted_idx = np.argsort(reward)[-int(N * 0.20):]

        elite_population = [population[idx] for idx in sorted_idx]
        elite_reward = [reward[idx] for idx in sorted_idx]

        theta_mean = np.mean(elite_population, axis=0)
        theta_sd = np.std(elite_population, axis=0)

        avg_reward = np.mean(elite_reward)
        print("Reward: {}".format(avg_reward))

        if avg_reward == 200:
            print("Game Cleared")
            break

    env = gym.wrappers.Monitor(env, "gym-results/", force=True)
    best_parm = elite_population[-1]

    for i in range(100):
        reward = run_episode(env, best_parm)
        print(reward)


if __name__ == '__main__':
    main()
