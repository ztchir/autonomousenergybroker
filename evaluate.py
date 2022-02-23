import os
import numpy as np
import imageio


def evaluate(env, model, num_steps=1000, test = ''):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    tariff_filenames = []
    profit_filenames = []
    cust_filenames = []   
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
              # create file name and append it to a list
        tariff, profit, cust = env.render()
        tariff_filename = f'tariff{i}.png'
        profit_filename = f'profit{i}.png'
        cust_filename = f'cust{i}.png'
        tariffgif = f'tariff{test}.gif'
        custgif = f'custgif{test}.gif'
        profitgif = f'profit{test}.gif'


        tariff_filenames.append(tariff_filename)
        profit_filenames.append(profit_filename)
        cust_filenames.append(cust_filename)
        # save frame
        tariff.savefig(os.path.join('./video/', tariff_filename))
        profit.savefig(os.path.join('./video/', profit_filename))
        cust.savefig(os.path.join('./video/', cust_filename))
        obs, reward, done, info = env.step(action)
        
        # Stats
        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
    
    # build gif
    with imageio.get_writer(os.path.join('./video/', tariffgif), mode='I') as writer:
        for filename in tariff_filenames:
            image = imageio.imread(os.path.join('./video/', filename))
            writer.append_data(image)
            os.remove(os.path.join('./video/', filename))
            
    # Remove files
    #for filename in set(tariff_filenames):
    #    os.remove(os.path.join('./video/', filename))


    # build gif
    with imageio.get_writer(os.path.join('./video/', profitgif), mode='I') as writer:
        for filename in profit_filenames:
            image = imageio.imread(os.path.join('./video/', filename))
            writer.append_data(image)
            os.remove(os.path.join('./video/', filename))
            
    # Remove files
    #for filename in set(profit_filenames):
    #    os.remove(os.path.join('./video/', filename))

    # build gif
    with imageio.get_writer(os.path.join('./video/', custgif), mode='I') as writer:
        for filename in cust_filenames:
            image = imageio.imread(os.path.join('./video/', filename))
            writer.append_data(image)
            os.remove(os.path.join('./video/', filename))
            
    # Remove files
    #for filename in set(cust_filenames):
    #    os.remove(os.path.join('./video/', filename))

    return mean_100ep_reward