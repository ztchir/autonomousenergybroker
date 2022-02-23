import gym
import json
import datetime as dt

#from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C


from env.EnergyBrokerEnv import EnergyBrokerEnv

import pandas as pd

df = pd.read_csv('./data/PUB_PriceHOEPPredispOR_2020.csv')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: EnergyBrokerEnv(df)])
model = A2C('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(50):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

