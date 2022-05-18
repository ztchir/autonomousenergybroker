import os
from subprocess import call

import gym
import json
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3 import PPO
from stable_baselines3 import DDPG


from env.monitor_training import SaveOnBestTrainingRewardCallback
from env.EnergyBrokerEnv import EnergyBrokerEnv
from evaluate import evaluate

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=100)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig(log_dir + "training.pdf")


df = pd.read_csv('./data/PUB_PriceHOEPPredispOR_2020.csv')

log_dir = "/home/ztchir/dev/autonomousenergybroker/logs/"
os.makedirs(log_dir, exist_ok=True)

# Evaluate Untrained model

#broker_env = DummyVecEnv([lambda: EnergyBrokerEnv(df)])
broker_env = Monitor(EnergyBrokerEnv(df, continuous=False), log_dir)

# callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir)

callback = EvalCallback(broker_env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=100,
                             deterministic=True, render=False)

model = A2C('MultiInputPolicy', broker_env, verbose=1, device='cuda')
#model = DDPG('MultiInputPolicy', broker_env, verbose=1, device='cuda')

mean_reward_before_train = evaluate(broker_env, model, num_steps=5000, test='untrained', gif=True)
# Train model
model.learn(total_timesteps=1000, callback=callback)
#
#results_plotter.plot_results([log_dir], 1000, results_plotter.X_TIMESTEPS, "A2C Energy Broker")
plot_results(log_dir)

#model.save('A2CEnergy')
#del model  # delete trained model to demonstrate loading

#model = A2C.load('A2CEnergy')
mean_reward_after_training = evaluate(broker_env, model, num_steps=5000, test='trained', gif=True)
del model
#print(mean_reward_before_train)
#print(mean_reward_after_training)
