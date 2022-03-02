import gym
import json
import datetime as dt
import pandas as pd

from knockknock import email_sender

#from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from env.EnergyBrokerEnv import EnergyBrokerEnv
from evaluate import evaluate

df = pd.read_csv('./data/PUB_PriceHOEPPredispOR_2020.csv')

'''
# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: EnergyBrokerEnv(df)])
model = A2C('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
print('#' * 30)
print('Initial:')
print('#' * 30)
env.render()
for i in range(600):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    print('#' * 30)
    env.render()
'''



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