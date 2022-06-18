import os
import gym
import json
import datetime as dt
import pandas as pd
import numpy as np

import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw
import collections

#from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from env.EnergyBrokerEnv import EnergyBrokerEnv

df = pd.read_csv('./data/PUB_PriceHOEPPredispOR_2020.csv')

env = DummyVecEnv([lambda: EnergyBrokerEnv(df)])
HOLD = np.array([[0, 0, 0]])
env.reset()

tariff_filenames = []
profit_filenames = []
cust_filenames = []

for i in range (600):
  env.step(HOLD)
  tariff, profit, cust = env.render()
  tariff_filename = f'tariff{i}.png'
  profit_filename = f'profit{i}.png'
  cust_filename = f'cust{i}.png'
  tariff_filenames.append(tariff_filename)
  profit_filenames.append(profit_filename)
  cust_filenames.append(cust_filename)
  # save frame
  tariff.savefig(os.path.join('./drive/MyDrive/ColabNotebooks/ECE-720_RL_Topics/Energy-Trading-Environment/video/', tariff_filename))
  profit.savefig(os.path.join('./drive/MyDrive/ColabNotebooks/ECE-720_RL_Topics/Energy-Trading-Environment/video/', profit_filename))
  cust.savefig(os.path.join('./drive/MyDrive/ColabNotebooks/ECE-720_RL_Topics/Energy-Trading-Environment/video/', cust_filename))


# build gif
with imageio.get_writer(os.path.join('./drive/MyDrive/ColabNotebooks/ECE-720_RL_Topics/Energy-Trading-Environment/video/', 'tariff.gif'), mode='I') as writer:
    for filename in tariff_filenames:
      image = imageio.imread(os.path.join('./drive/MyDrive/ColabNotebooks/ECE-720_RL_Topics/Energy-Trading-Environment/video/', tariff_filename))
      writer.append_data(image)
        
    # Remove files
for filename in set(tariff_filenames):
    os.remove(os.path.join('./drive/MyDrive/ColabNotebooks/ECE-720_RL_Topics/Energy-Trading-Environment/video/', filename))


# build gif
with imageio.get_writer(os.path.join('./drive/MyDrive/ColabNotebooks/ECE-720_RL_Topics/Energy-Trading-Environment/video/', 'profit.gif'), mode='I') as writer:
    for filename in profit_filenames:
      image = imageio.imread(os.path.join('./drive/MyDrive/ColabNotebooks/ECE-720_RL_Topics/Energy-Trading-Environment/video/', profit_filename))
      writer.append_data(image)
        
    # Remove files
for filename in set(profit_filenames):
    os.remove(os.path.join('./drive/MyDrive/ColabNotebooks/ECE-720_RL_Topics/Energy-Trading-Environment/video/', filename))

# build gif
with imageio.get_writer(os.path.join('./drive/MyDrive/ColabNotebooks/ECE-720_RL_Topics/Energy-Trading-Environment/video/', 'cust.gif'), mode='I') as writer:
    for filename in cust_filenames:
      image = imageio.imread(os.path.join('./drive/MyDrive/ColabNotebooks/ECE-720_RL_Topics/Energy-Trading-Environment/video/', cust_filename))
      writer.append_data(image)
        
    # Remove files
for filename in set(cust_filenames):
    os.remove(os.path.join('./drive/MyDrive/ColabNotebooks/ECE-720_RL_Topics/Energy-Trading-Environment/video/', filename))


'''
  frame, profit, cust = env.render()
  filename = f'{i}.png'
  filenames.append(filename)
    
  # save frame
  frame.savefig(os.path.join('./drive/MyDrive/ColabNotebooks/ECE-720_RL_Topics/Energy-Trading-Environment/video/', filename))
  #plt.close()

# build gif
with imageio.get_writer(os.path.join('./drive/MyDrive/ColabNotebooks/ECE-720_RL_Topics/Energy-Trading-Environment/video/', 'hold.gif'), mode='I') as writer:
    for filename in filenames:
      image = imageio.imread(os.path.join('./drive/MyDrive/ColabNotebooks/ECE-720_RL_Topics/Energy-Trading-Environment/video/', filename))
      writer.append_data(image)
        
    # Remove files
for filename in set(filenames):
    os.remove(os.path.join('./drive/MyDrive/ColabNotebooks/ECE-720_RL_Topics/Energy-Trading-Environment/video/', filename))
'''