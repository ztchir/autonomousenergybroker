import gym
from gym import spaces
import pandas as pd
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import collections


# Add ability to have more than one broker later
MAX_ACCOUNT_BALANCE = 2147483647
MAX_ENERGY_PRICE = 1
MAX_TARIFF = 10
MAX_STEPS = 20000

# Start with 6 customers
#np.random.seed(1001)
NUM_BROKERS = 5
NUM_CUSTOMERS = 1000
#WHOLESALE = np.array([0.07 for i in range(NUM_BROKERS)])
WHOLESALE = np.array([0.07 for i in range(1)])

CUSTOMER = np.arange(0, NUM_CUSTOMERS)
CUSTOMER_TEMP = 2 # With a Customer Temp of 2 there is a 67% chance Customers will choose top 3 tariffs
#CUSTOMER_TEMP = np.random.randint(0,100000000000, NUM_CUSTOMERS)
#CUSTOMER_TEMP = 100000000 * np.ones(NUM_CUSTOMERS)
#CUSTOMER_TEMP = [1, 1, 1, np.inf, np.inf, np.inf]
#CUSTOMER_TEMP = [1, 1, 1, 1, 1, 1] # Customers should choose the cheapest tariff
CUSTOMER_BASE = np.random.randint(0, NUM_BROKERS, NUM_CUSTOMERS)
#CUSTOMER_BASE = [1, 1, 1, 1, 1, 1] # All customers assigned to broker 2 initially
INITIAL_ACCOUNT_BALANCE = np.full(1, 1000000)
INITIAL_BROKER_VOLUME = np.zeros(NUM_BROKERS)
INITIAL_TARIFFS = 0.166 * np.random.rand(NUM_BROKERS) # using normal distribution around average tariff price $0.166kW/h
#INITIAL_TARIFFS = [.166, 1.00] # Broker 1 starts with a cheaper price

# MINIMUM TARIFF and MAXIMUM TARIFF
MINIMUM_TARIFF = 0.1
MAXIMUM_TARIFF = 10

class EnergyBrokerEnv(gym.Env):
    """An Energy Broker Environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(EnergyBrokerEnv, self).__init__()

        self.df = df
        self.reward_range = (-MAX_ACCOUNT_BALANCE, MAX_ACCOUNT_BALANCE)

        # Actions of the broker Increase %x, Decrease %x, or Hold
        self.act_space = spaces.Box(low=-0.1, high=0.1, shape=(1,)) # Brokers limited to 1/10 or doubling in a single timestep
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
            #'wholesale_space' : gym.spaces.Box(low=0, high=1, shape=(11,)),   # Remove this for now to simplify the case
            'tariff_space' : gym.spaces.Box(low=0, high=MAX_TARIFF, shape=(NUM_BROKERS,)),
            #'balance_space' : gym.spaces.Box(low=0, high=MAX_ACCOUNT_BALANCE, shape=(NUM_BROKERS,)),
            'balance_space' : gym.spaces.Box(low=0, high=MAX_ACCOUNT_BALANCE, shape=(1,)),
        }
        self.observation_space = spaces.Dict(self.obs_spaces)
        

    # Define historical look back length
    #### This will change when adding broker preditcion ####

    def _next_observation(self):
        #frame = np.array([
        #    self.df.loc[self.current_step: self.current_step + 10, 'OR 30 Min'].values / MAX_ENERGY_PRICE,
        #])
        #### Determine How to add consumption
        self.broker_volume = copy(INITIAL_BROKER_VOLUME)
        self.cust_volume = copy(INITIAL_BROKER_VOLUME)

        self.cust_volume = (600 / 3600*30) * (1 + np.random.normal(0, 0.25, NUM_CUSTOMERS)) # average hourly consumption
        for base in range(0, NUM_CUSTOMERS):
            self.broker_volume[self.cust_base[base]] += self.cust_volume[base]        
            self.pred_volume[self.cust_base[base]] += self.cust_volume[base] * (1 + np.random.normal(0, 0.25))
        self.profit = self.tariff[0] * self.broker_volume[0] - self.pred_volume[0] * WHOLESALE
        np.add(self.balance, self.profit, out=self.balance, casting='unsafe')       
    
        obs = {
            'customer_base' : self.cust_base,
            #'wholesale_space': frame,
            'tariff_space' : self.tariff,
            'balance_space' : self.balance,
        }

        return obs

    def _take_action(self, action):
        self.action_taken = action
        # Broker 0 is the broker we are concerened with here. The other brokers
        # are randomly controlled
        # for i in range(len(action)):
        # Temporary addition of random action of broker 1 and 2
        self.tariff[0] *= (1 + action[0]) #######
        for i in range(1, NUM_BROKERS):
            self.tariff[i] *= (1 + np.random.uniform(-0.1,0.1))
  
        for i in range(NUM_BROKERS):
            if self.tariff[i] < MINIMUM_TARIFF:
              self.tariff[i] = MINIMUM_TARIFF
            elif self.tariff[i] > MAXIMUM_TARIFF:
              self.tariff[i] = MAXIMUM_TARIFF

        rank = np.argsort(self.tariff) + 1

        self.cust_base = []
        '''
        for customer in self.customers:
            prob = []
            ranksum = np.sum(np.exp(-rank/self.temp[customer]))
            for i in rank:
                prob.append(-np.exp(-i/self.temp[customer]) / ranksum)
        '''
        for customer in self.customers:
            prob = []
            ranksum = np.sum(np.exp(-rank/self.temp))
            for i in rank:
                prob.append(np.exp(-i/self.temp) / ranksum)
            self.cust_base = np.random.choice([0, 1, 2, 3, 4], p=prob, size=NUM_CUSTOMERS)
            '''
            options = np.where(prob == np.min(prob))
            if len(options[0]) > 1:
              self.cust_base.append(np.random.choices(options[0]))
            else:
              self.cust_base.append(options[0][0])
            '''

    def step(self, action):
        # Execute one time step within the enviroment
        # print('Action=', action)
        self._take_action(action)

        self.current_step += 1
        
        if self.current_step > len(self.df.loc[:].values) - 11:
            self.current_step = 0
        
        delay_modifier = (self.current_step / MAX_STEPS)
    
        self.step_profit.append(self.balance)

        self.tariff_data1.append(self.tariff[0])
        self.tariff_data2.append(self.tariff[1])
        self.tariff_data3.append(self.tariff[2])
        self.tariff_data4.append(self.tariff[3])
        self.tariff_data5.append(self.tariff[4])

        reward = self.profit * delay_modifier
        reward_as_float = reward.astype(float)
        #reward = self.balance * delay_modifier
        #done = np.any(self.balance <= INITIAL_BROKER_VOLUME) # Note  INITIAL_BROKER_VOLUME = 0 this is just zero
        done = np.any(self.balance <= 10000) # Note  INITIAL_BROKER_VOLUME = 0 this is just zero
        
        obs = self._next_observation()

        return obs, reward_as_float, done, 1.05236

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
        plt.title('Broker Profit')
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
        