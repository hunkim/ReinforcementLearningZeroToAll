# Evolution Strategies BipedalWalker-v2
# https://blog.openai.com/evolution-strategies/
# converges at around iter 100 in 5 minutes
# for testing model set reload=True

import gym
import numpy as np
import cPickle as pickle
import sys

env = gym.make('BipedalWalker-v2')
np.random.seed(10)

hl_size = 100
version = 1
npop = 50
sigma = 0.1
alpha = 0.03
iter_num = 300
aver_reward = None
allow_writing = True
reload = False

print(hl_size, version, npop, sigma, alpha, iter_num)

if reload:
    model = pickle.load(open('model-pedal%d.p' % version, 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(24, hl_size) / np.sqrt(24)
    model['W2'] = np.random.randn(hl_size, 4) / np.sqrt(hl_size)

def get_action(state, model):
    hl = np.matmul(state, model['W1'])
    hl = np.tanh(hl)
    action = np.matmul(hl, model['W2'])
    action = np.tanh(action)

    return action

def f(model, render=False):
    state = env.reset()
    total_reward = 0
    for t in range(iter_num):
        if render: env.render()

        action = get_action(state, model)
        state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break
    return total_reward

if reload:
    iter_num = 10000
    for i_episode in range(10):
        print(f(model, True))
    sys.exit('demo finished')

for i in range(10001):
    N = {}
    for k, v in model.iteritems():
        N[k] = np.random.randn(npop, v.shape[0], v.shape[1])
    R = np.zeros(npop)

    for j in range(npop):
        model_try = {}
        for k, v in model.iteritems():
            model_try[k] = v + sigma*N[k][j]
        R[j] = f(model_try)

    A = (R - np.mean(R)) / np.std(R)
    for k in model:
        model[k] = model[k] + alpha/(npop*sigma) * np.dot(N[k].transpose(1, 2, 0), A)

    cur_reward = f(model)
    aver_reward = aver_reward * 0.9 + cur_reward * 0.1 if aver_reward != None else cur_reward
    print('iter %d, cur_reward %.2f, aver_reward %.2f' % (i, cur_reward, aver_reward))

    if i % 10 == 0 and allow_writing:
        pickle.dump(model, open('model-pedal%d.p' % version, 'wb'))
