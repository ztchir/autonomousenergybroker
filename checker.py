from stable_baselines3.common.env_checker import check_env
from env.EnergyBrokerEnv import EnergyBrokerEnv
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np
import pandas as pd

df = pd.read_csv('./data/PUB_PriceHOEPPredispOR_2020.csv')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: EnergyBrokerEnv(df)])

obs = env.reset()
env.render()
print('#' * 20)
print('Enviroment Spaces')
print('#' * 20)
print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())
print('#' * 20)



############# Test Customer Behaviour #############
# Here the brokers hold thier prices this will show how the customers behave
print('#' * 20)
print('Test Customer behaviour on holding tariff steady')
print('#' * 20)
print('Initial State')
env.render()
HOLD = np.array([[0, 0]])
n_steps = 1
for step in range(n_steps):
    print("Step {}".format(step + 1))
    obs, reward, done, info = env.step(HOLD)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    env.render()
    if done:
        print("Goal reached!", 'reward=', reward)
        break
print('#' * 20)


############# Test Customer Behaviour #############
# Here the broker 1 will increase tariff 0.05 every step and once exceeds broker 2 customers will switch back
# Test that the tariffs are being adjusted
print('#' * 20)
print('Test Customer Change to lower tariff')
print('#' * 20)
env.reset()
print('Initial State')
env.render()
BROKER_1_INC = np.array([[0.1, 0]])
n_steps = 20
for step in range(n_steps):
    print("Step {}".format(step + 1))
    obs, reward, done, info = env.step(BROKER_1_INC)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    env.render()
    if done:
        print("Goal reached!", 'reward=', reward)
        break
print('#' * 20)

