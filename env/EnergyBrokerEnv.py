import random
import json
import gym
from gym import spaces
from gym.core import ObservationWrapper
import pandas as pd
import numpy as np

# Add ability to have more than one broker later
MAX_ACCOUNT_BALANCE = 2147483647
MAX_ENERGY_PRICE = 1
MAX_TARIFF = 1
MAX_STEPS = 20000

# Start with 6 customers
#np.random.seed(1001)
NUM_BROKERS = 4
NUM_CUSTOMERS = 6
WHOLESALE = np.array([0.07 for i in range(NUM_BROKERS)])
CUSTOMER = np.arange(0, NUM_CUSTOMERS)
#CUSTOMER_TEMP = np.random.randint(0,10000000, NUM_CUSTOMERS)
CUSTOMER_TEMP = [0, 1, 500000, 100000000000, 10000000000000, 100000000000000]
CUSTOMER_BASE = np.random.randint(0, NUM_BROKERS, NUM_CUSTOMERS)
INITIAL_ACCOUNT_BALANCE = np.full(NUM_BROKERS, 1000000)
INITIAL_BROKER_VOLUME = np.zeros(NUM_BROKERS)
INITIAL_TARIFFS = 0.166 * np.random.rand(NUM_BROKERS) # using normal distribution around average tariff price $0.166kW/h

class EnergyBrokerEnv(gym.Env):
    """An Energy Broker Environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(EnergyBrokerEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the broker Increase %x, Decrease %x, or Hold
        self.act_space = spaces.Box(low=-1, high =1, shape=(NUM_BROKERS,)) # Brokers limited to 1/2 or doubling in a single timestep
        #{
            #'action_types' : spaces.MultiDiscrete(np.full(NUM_BROKERS, 3)),
            #'amount' : spaces.Box(low=0, high=1, shape=(NUM_BROKERS,))
        #}
        self.action_space = self.act_space
        
        # Observations contain Historical Wholesale price, number of customers,
        # and current account balance

        #### Tariffs of other brokers to be added later ####
        #### This will change when adding broker preditcion ####
        self.obs_spaces = {
            'customer_base' : gym.spaces.Box(low=0, high=NUM_BROKERS, shape=(NUM_CUSTOMERS,)),
            'wholesale_space' : gym.spaces.Box(low=0, high=1, shape=(11,)),
            'tariff_space' : gym.spaces.Box(low=0, high=MAX_TARIFF, shape=(NUM_BROKERS,)),
            'balance_space' : gym.spaces.Box(low=0, high=MAX_ACCOUNT_BALANCE, shape=(NUM_BROKERS,)),
        }
        self.observation_space = spaces.Dict(self.obs_spaces)
        

    # Define historical look back length
    #### This will change when adding broker preditcion ####

    def _next_observation(self):
        frame = np.array([
            self.df.loc[self.current_step: self.current_step + 10, 'OR 30 Min'].values / MAX_ENERGY_PRICE,
        ])
        #### Determine How to add consumption
        self.broker_volume = INITIAL_BROKER_VOLUME
        self.cust_volume = INITIAL_BROKER_VOLUME

        self.cust_volume = (600 / 3600*30) * 1 + np.random.normal(0, 0.25, NUM_CUSTOMERS) # average hourly consumption
        for base in range(0, NUM_CUSTOMERS):
            self.broker_volume[self.cust_base[base]] += self.cust_volume[base]        
            self.pred_volume[self.cust_base[base]] += self.cust_volume[base] * (1 + np.random.normal(0, 0.25))
        np.add(self.balance, self.tariff * self.broker_volume - self.pred_volume * WHOLESALE, out=self.balance, casting='unsafe')       
    

        obs = {
            'customer_base' : self.cust_base,
            'wholesale_space': frame,
            'tariff_space' : self.tariff,
            'balance_space' : self.balance,
        }

        return obs

    def _take_action(self, action):
        '''
        action_type = action['action_type']
        amount = action['amount']
        self.action_taken = action
        for i in range(action_type):
            if action_type[i] < 1:
                # Increase amount % of max
                self.tariff[i] *= (1 + amount[i])

            elif action_type[i] < 2:
                # Decrease amount % of max
                self.tariff[i] *= (1 - amount[i])
        '''
        self.action_taken = action

        for i in range(len(action)):
            self.tariff[i] *= (1 + action[i]) #######

        rank = np.argsort(self.tariff) + 1


        self.cust_base = []
        for customer in self.customers:
            prob = []
            ranksum = np.sum(np.exp(-rank/self.temp[customer]))
            for i in rank:
                prob.append(-np.exp(-i/self.temp[customer]) / ranksum)
            self.cust_base.append(np.argmax(prob))


    def step(self, action):
        # Execute one time step within the enviroment
        self._take_action(action)

        self.current_step += 1
        
        if self.current_step > len(self.df.loc[:].values) - 11:
            self.current_step = 0
        
        delay_modifier = (self.current_step / MAX_STEPS)
    

        reward = self.balance[0] * delay_modifier
        done = np.any(self.balance <= INITIAL_BROKER_VOLUME) # Note  INITIAL_BROKER_VOLUME = 0 this is just zero
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.customers = CUSTOMER
        self.cust_base = CUSTOMER_BASE
        self.tariff = INITIAL_TARIFFS
        self.broker_volume = INITIAL_BROKER_VOLUME
        self.pred_volume = INITIAL_BROKER_VOLUME
        self.temp = CUSTOMER_TEMP

        self.current_step = np.random.randint(
            0, len(self.df.loc[:, 'OR 30 Min'].values) - 11)

        return self._next_observation()



    def render(self, mode='human', close=False):
        # Render the environment to the screen
        self.profit = self.balance - INITIAL_ACCOUNT_BALANCE

        np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

        print(f'Step: {self.current_step}')
        print(f'Action Taken: {self.action_taken}')
        print(f'Customer Base: {self.cust_base}')
        print(f'Broker Volume: {self.broker_volume}')
        print(f'Pred Volume: {self.pred_volume}')
        print(f'Tariffs: {self.tariff}')
        print(f'Balance: {self.balance}')
        print(f'Profit: {self.profit}')