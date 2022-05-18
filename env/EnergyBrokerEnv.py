#from multiprocessing.pool import INIT
import gym
from gym import spaces
import pandas as pd
# import cupy as np
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import collections


# Add ability to have more than one broker later
MAX_ACCOUNT_BALANCE = 2147483647
MAX_TARIFF = 10
MAX_STEPS = 2000000

# Start with 6 customers
np.random.seed(1001)
NUM_BROKERS = 5
NUM_CUSTOMERS = 10000
WHOLESALE = np.array([0.1 for i in range(1)]) # Albert Wholesale Price ~$103$/MWh or 0.103$/kWh

CUSTOMER = np.arange(0, NUM_CUSTOMERS)
CUSTOMER_TEMP = 0.1 # With a Customer Temp of 2 there is a 67% chance Customers will choose top 3 tariffs
CUSTOMER_BASE = np.random.randint(0, NUM_BROKERS, NUM_CUSTOMERS)
INITIAL_ACCOUNT_BALANCE = np.full(1, 0)
INITIAL_BROKER_VOLUME = np.zeros(NUM_BROKERS)

# MINIMUM TARIFF and MAXIMUM TARIFF
BROKER_CHANGE_LIMIT = 0.5 # LIMITED TO MAXIMUM CHANGE 50%
MINIMUM_TARIFF = 0.0 # Minimum 0.0 cents/kWh
MAXIMUM_TARIFF = 0.20 # Maximum 0.15 cents/kWh

#INITIAL_TARIFFS = np.full(NUM_BROKERS, 0.05)
INITIAL_TARIFFS = np.array([0.15, 0.05, 0.1, 0.07, 0.19])
#INITIAL_TARIFFS = MAXIMUM_TARIFF * np.random.rand(NUM_BROKERS) # using normal distribution around average tariff price $0.166kW/h



class EnergyBrokerEnv(gym.Env):
    """An Energy Broker Environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df, 
        continuous: bool = False,
        ):

        super(EnergyBrokerEnv, self).__init__()

        self.df = df
        self.continuous = continuous
        self.reward_range = (-MAX_ACCOUNT_BALANCE, MAX_ACCOUNT_BALANCE)

        # Actions of the broker Increase %x, Decrease %x, or Hold
        if self.continuous:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        else:
            self.action_space = spaces.Discrete(3)
        #{
            #'action_types' : spaces.MultiDiscrete(np.full(NUM_BROKERS, 3)),
            #'amount' : spaces.Box(low=0, high=1, shape=(NUM_BROKERS,))
        #}
        
        # Observations contain Historical Wholesale price, number of customers,
        # and current account balance

        #### Tariffs of other brokers to be added later ####
        #### This will change when adding broker preditcion ####
        self.obs_spaces = {
            'market_share' : gym.spaces.Box(low=0, high=1, shape=(NUM_BROKERS,)),
            #'wholesale_space' : gym.spaces.Box(low=0, high=1, shape=(11,)),   # Remove this for now to simplify the case
            'tariff_space' : gym.spaces.Box(low=0, high=1, shape=(NUM_BROKERS,)),
            #'balance_space' : gym.spaces.Box(low=0, high=MAX_ACCOUNT_BALANCE, shape=(NUM_BROKERS,)),
            'balance_space' : gym.spaces.Box(low=0, high=1, shape=(1,)),
        }
        self.observation_space = spaces.Dict(self.obs_spaces)
        

    # Define historical look back length
    #### This will change when adding broker preditcion ####

    def _next_observation(self):
        #frame = np.array([
        #    self.df.loc[self.current_step: self.current_step + 10, 'OR 30 Min'].values / MAX_ENERGY_PRICE,
        #])       
        
        # Normalize the observation space
        obs = {
            'market_share' : self.customer_share / NUM_CUSTOMERS,
            #'wholesale_space': frame,
            'tariff_space' : (self.tariff - MINIMUM_TARIFF) / (MAXIMUM_TARIFF - MINIMUM_TARIFF),
            'balance_space' : self.balance / MAX_ACCOUNT_BALANCE,
        }

        return obs

    def _take_action(self, action):
        # Broker 0 is the broker we are concerened with here. The other brokers
        # are randomly controlled
        # for i in range(len(action)):
        # Temporary addition of random action of broker 1 and 2
        if self.continuous:
           self.tariff[0] *= (1 + action[0] * MAXIMUM_TARIFF)
           #self.tariff[0] = action[0] * MAXIMUM_TARIFF
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "

        if not self.continuous and action == 1:    
            self.tariff[0] += 0.005
        elif not self.continuous and action == 2:
            self.tariff[0] -= 0.005

        #if self.current_step == 500:
        #    self.tariff[1] = 0.025
        ########
        # ADD Continuous and Discrete actitons similar to LunarLander-V2
        ########
        '''
        if self.action_taken == 1:
            self.tariff[0] = (self.tariff[0] + 0.01)
        elif self.action_taken == 2:
            self.tariff[0] = (self.tariff[0] - 0.01)
        '''
        # Random changes in other broker tariffs
        # for i in range(1, NUM_BROKERS):
        #     self.tariff[i] *= (1 + np.random.uniform(-0.05,0.05))

        self.tariff = np.clip(self.tariff, MINIMUM_TARIFF, MAXIMUM_TARIFF).astype(np.float32)

        rank = np.argsort(self.tariff) + 1

        self.cust_base = []
        
        # Customers choose new broker
        prob = []
        ranksum = np.sum(np.exp(-rank/self.temp))
        for i in rank:
            prob.append(np.exp(-i/self.temp) / ranksum)
        self.cust_base = np.random.choice(list(range(NUM_BROKERS)), p=prob, size=NUM_CUSTOMERS)

        #### Determine How to add consumption
        self.customer_share = copy(INITIAL_BROKER_VOLUME) # Zeros market share for each broker
        self.broker_volume = copy(INITIAL_BROKER_VOLUME) # Initial Volume is vector of zeros for NUM_BROKERS
        self.cust_volume = copy(INITIAL_BROKER_VOLUME) 

        self.cust_volume = (600) * (1 + np.random.normal(0, 0, NUM_CUSTOMERS)) # average hourly consumption 600kWh
        for base in range(0, NUM_CUSTOMERS):
            self.customer_share[self.cust_base[base]] += 1
            self.broker_volume[self.cust_base[base]] += self.cust_volume[base]        
            #self.pred_volume[self.cust_base[base]] += self.cust_volume[base] * (1 + np.random.normal(0, 0.01))
        self.profit = self.tariff[0] * self.broker_volume[0] - self.broker_volume[0] * WHOLESALE # They always buy the right amount of energy
        np.add(self.balance, self.profit, out=self.balance, casting='unsafe')


    def step(self, action):
        # Execute one time step within the enviroment
        # print('Action=', action)
        self._take_action(action)

        self.current_step += 1
        
        if self.current_step > len(self.df.loc[:].values) - 11:
            self.current_step = 0
        
        delay_modifier = (self.current_step / MAX_STEPS)
    
        self.step_profit.append(self.balance[0])

        self.tariff_data1.append(self.tariff[0])
        self.tariff_data2.append(self.tariff[1])
        self.tariff_data3.append(self.tariff[2])
        self.tariff_data4.append(self.tariff[3])
        self.tariff_data5.append(self.tariff[4])

        reward = self.profit[0]
        reward_as_float = reward.astype(float)
        done = self.current_step > 20000 
        obs = self._next_observation()

        return obs, reward_as_float, done, {}

    def reset(self):
        self.tariff_data1 = []
        self.tariff_data2 = []
        self.tariff_data3 = []
        self.tariff_data4 = []
        self.tariff_data5 = []
        self.step_profit = []

        self.balance = copy(INITIAL_ACCOUNT_BALANCE)
        self.customers = copy(CUSTOMER)
        self.cust_base = copy(CUSTOMER_BASE)
        self.customer_share = copy(INITIAL_BROKER_VOLUME)
        self.tariff = copy(INITIAL_TARIFFS)
        self.broker_volume = copy(INITIAL_BROKER_VOLUME)
        self.pred_volume = copy(INITIAL_BROKER_VOLUME)
        self.temp = copy(CUSTOMER_TEMP)

        self.current_step = np.random.randint(
            0, len(self.df.loc[:, 'OR 30 Min'].values) - 20)

        return self._next_observation()



    def render(self, mode='human', close=False):
        # Render the environment to the screen
        #self.profit = self.balance - INITIAL_ACCOUNT_BALANCE

        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

        brokers = ['Broker 1', 'Broker 2', 'Broker 3', 'Broker 4', 'Broker 5']

        tariff_fig = plt.figure() 
        plt.plot(self.tariff_data1)
        plt.plot(self.tariff_data2)
        plt.plot(self.tariff_data3)
        plt.plot(self.tariff_data4)
        plt.plot(self.tariff_data5)
        plt.title('Broker Tariffs')
        plt.ylabel('$/kWh')
        plt.xlabel('Time Step')
        plt.legend(brokers)
        plt.close()

        profit_fig = plt.figure()
        plt.plot(self.step_profit)
        plt.title('Broker Account Balance')
        plt.xlabel('Time Step')
        plt.ylabel('$')
        plt.legend(brokers)
        plt.close()

        customer_fig = plt.figure()
        ax = customer_fig.add_axes([0,0,1,1])
        ax.axis('equal')
        counts = collections.Counter(self.cust_base)
        customers = [counts[0], counts[1], counts[2], counts[3], counts[4]]
        ax.pie(customers, labels = brokers, autopct='%1.2f%%')
        plt.close()
        return tariff_fig, profit_fig, customer_fig
        
