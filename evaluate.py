import os
import numpy as np
import imageio




def evaluate(env, model, num_steps=1000, test = '', run_dir = '', gif=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """

    plots_dir = "/home/ztchir/dev/autonomousenergybroker/" + run_dir + "/plot/"
    os.makedirs(plots_dir, exist_ok=True)

    tariff_filenames = []
    profit_filenames = []
    cust_filenames = []   
    episode_rewards = [0.0]
    obs = env.reset()

    tariffgif = f'tariff{test}.gif'
    custgif = f'custgif{test}.gif'
    profitgif = f'balance{test}.gif'
    tariffplot = f'tariff{test}.png'
    custplot = f'custgif{test}.png'
    profitplot = f'balance{test}.png'

    if gif == True:
        video_dir = "/home/ztchir/dev/autonomousenergybroker/" + run_dir + "/video/"
        os.makedirs(video_dir, exist_ok=True)
        for i in range(num_steps):
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=False)
                # create file name and append it to a list
            tariff, profit, cust = env.render()
            tariff_filename = f'tariff{i}.png'
            profit_filename = f'balance{i}.png'
            cust_filename = f'cust{i}.png'



            tariff_filenames.append(tariff_filename)
            profit_filenames.append(profit_filename)
            cust_filenames.append(cust_filename)
            # save frame
            tariff.savefig(os.path.join(video_dir, tariff_filename))
            profit.savefig(os.path.join(video_dir, profit_filename))
            cust.savefig(os.path.join(video_dir, cust_filename))
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
        with imageio.get_writer(os.path.join(video_dir, tariffgif), mode='I') as writer:
            for filename in tariff_filenames:
                image = imageio.imread(os.path.join(video_dir, filename))
                writer.append_data(image)
                os.remove(os.path.join(video_dir, filename))
                


        # build gif
        with imageio.get_writer(os.path.join(video_dir, profitgif), mode='I') as writer:
            for filename in profit_filenames:
                image = imageio.imread(os.path.join(video_dir, filename))
                writer.append_data(image)
                os.remove(os.path.join(video_dir, filename))
                

    # build gif
        with imageio.get_writer(os.path.join(video_dir, custgif), mode='I') as writer:
            for filename in cust_filenames:
                image = imageio.imread(os.path.join(video_dir, filename))
                writer.append_data(image)
                os.remove(os.path.join(video_dir, filename))
                

    else:
        for i in range(num_steps):
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=True)
            # create file name and append it to a list
            # Stats
            obs, reward, done, info = env.step(action)
            episode_rewards[-1] += reward
            
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
            # Compute mean reward for the last 100 episodes

        mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
        print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    tariff, profit, cust = env.render()


    # save frame
    tariff.savefig(os.path.join(plots_dir, tariffplot))
    profit.savefig(os.path.join(plots_dir, profitplot))
    cust.savefig(os.path.join(plots_dir, custplot))

    return mean_100ep_reward