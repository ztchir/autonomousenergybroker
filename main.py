import os
import sys
from subprocess import call

import gym
import json
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from knockknock import email_sender

#from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
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
    weights = np.repeat(1.0, window) / window # np array of length window and value 1/window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, sub_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=10)
    # Truncate x
    x = x[len(x) - len(y):]

<<<<<<< HEAD
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig(log_folder + sub_folder + "training.pdf")


# Set descriptive folder name for multiple runs

# A breif description of the run, what changed what are we testing
test_description = '''

'''

run_dir = "test_validation_curve"

df = pd.read_csv('./data/PUB_PriceHOEPPredispOR_2020.csv')

log_dir = "/home/ztchir/dev/autonomousenergybroker/" + run_dir + "/logs/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(log_dir + 'training1/', exist_ok=True)
os.makedirs(log_dir + 'training2/', exist_ok=True)
log_dir1 = log_dir + 'training1/'
log_dir2 = log_dir + 'training2/'

broker_env = Monitor(EnergyBrokerEnv(df, continuous=False), log_dir)

stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=5, verbose=1)

callback1 = EvalCallback(broker_env, best_model_save_path=log_dir1,
                            log_path=log_dir1, eval_freq=500,
                            deterministic=True, render=False,
                            callback_after_eval=stop_train_callback)

callback2 = EvalCallback(broker_env, best_model_save_path=log_dir2,
                            log_path=log_dir2, eval_freq=500,
                            deterministic=True, render=False,
                            callback_after_eval=stop_train_callback)

agent = A2C('MultiInputPolicy', broker_env, verbose=1, device='cuda')
# Evaluate Untrained model
mean_reward_before_train = evaluate(broker_env, agent, num_steps=1000, test='untrained', run_dir=run_dir, gif=True)
# Train model
agent.learn(total_timesteps=int(1e5), callback=callback1)

# Plot learning curve
plot_results(log_dir, 'training1/')

# Evaluate trained model
mean_reward_after_training = evaluate(broker_env, agent, num_steps=1000, test='validation', run_dir=run_dir, gif=True)

agent.learn(total_timesteps=int(1e5), callback=callback2)

# Plot learning curve
plot_results(log_dir, 'training2/')

# Evaluate trained model
mean_reward_after_training = evaluate(broker_env, agent, num_steps=1000, test='trained', run_dir=run_dir, gif=True)

=======
# Evaluate Untrained model
@email_sender(recipient_emails=["ztchir@gmail.com"], sender_email="zachnotification@gmail.com")
def model_evaluation():
    broker_env = DummyVecEnv([lambda: EnergyBrokerEnv(df)])
    model = A2C('MultiInputPolicy', broker_env, verbose=1)
    mean_reward_before_train = evaluate(broker_env, model, num_steps=100, test='untrained')

    # Train model
    model.learn(total_timesteps=1000)
    model.save('A2CEnergy')
    del model  # delete trained model to demonstrate loading

    model = A2C.load('A2CEnergy')
    mean_reward_after_training = evaluate(broker_env, model, num_steps=100, test='trained')
    with open('results.txt', 'w') as f:
        f.write('Mean reward before training: ' + mean_reward_before_train.astype('str') + '\n')
        f.write('Mean reward after training: ' + mean_reward_after_training.astype('str') + '\n')

    print(mean_reward_before_train)
    print(mean_reward_after_training)

model_evaluation()
>>>>>>> a237baf8c8ab7e598279c2145d954877d25622af
