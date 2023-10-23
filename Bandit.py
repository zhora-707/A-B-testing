"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER

import logging

#logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
#logger = logging.getLogger(__name__)


from abc import ABC, abstractmethod
from logs import *

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig
logger = logging.getLogger("MAB Application")

# Define Bandit_Reward and num_trials
Bandit_Reward = [1, 2, 3, 4]
num_trials = 20000


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        self.true_means = p  # True means of each arm
        self.estimated_means = [0.0] * len(p)  # Estimated means of each arm
        self.action_counts = [0] * len(p)  # Number of times each arm is pulled


    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():
    def __init__(self):
        self.cumulative_rewards = {"Epsilon-Greedy": [], "Thompson Sampling": []}
        self.cumulative_regrets = {"Epsilon-Greedy": [], "Thompson Sampling": []}

    def plot1(self, epsilon_greedy_rewards, thompson_rewards):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(epsilon_greedy_rewards), label="Epsilon-Greedy")
        plt.plot(np.cumsum(thompson_rewards), label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(np.log(np.cumsum(epsilon_greedy_rewards)), label="Epsilon-Greedy (log scale)")
        plt.plot(np.log(np.cumsum(thompson_rewards)), label="Thompson Sampling (log scale)")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward (log scale)")
        plt.legend()

        plt.show()

    def plot2(self, epsilon_greedy_rewards, thompson_rewards):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(epsilon_greedy_rewards), label="Epsilon-Greedy")
        plt.plot(np.cumsum(thompson_rewards), label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()

        plt.subplot(1, 2, 2)
        cumulative_regret_epsilon = np.cumsum([max(Bandit_Reward) - r for r in epsilon_greedy_rewards])
        cumulative_regret_thompson = np.cumsum([max(Bandit_Reward) - r for r in thompson_rewards])
        plt.plot(cumulative_regret_epsilon, label="Epsilon-Greedy")
        plt.plot(cumulative_regret_thompson, label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Regret")
        plt.legend()

        plt.show()

    def store_rewards_to_csv(self, epsilon_greedy_rewards, thompson_rewards):
        data = {
            "Bandit": ["Epsilon-Greedy"] * len(epsilon_greedy_rewards) + ["Thompson Sampling"] * len(thompson_rewards),
            "Reward": epsilon_greedy_rewards + thompson_rewards,
            "Algorithm": ["Epsilon-Greedy"] * len(epsilon_greedy_rewards) + ["Thompson Sampling"] * len(thompson_rewards)
        }
        df = pd.DataFrame(data)
        df.to_csv("bandit_rewards.csv", index=False)

def report_cumulative_reward_and_regret(self, epsilon_greedy_rewards, thompson_rewards):
        cumulative_reward_epsilon = np.sum(epsilon_greedy_rewards)
        cumulative_reward_thompson = np.sum(thompson_rewards)
        cumulative_regret_epsilon = np.sum([max(Bandit_Reward) - r for r in epsilon_greedy_rewards])
        cumulative_regret_thompson = np.sum([max(Bandit_Reward) - r for r in thompson_rewards])

        print(f"Epsilon-Greedy Cumulative Reward: {cumulative_reward_epsilon:.2f}")
        print(f"Thompson Sampling Cumulative Reward: {cumulative_reward_thompson:.2f}")
        print(f"Epsilon-Greedy Cumulative Regret: {cumulative_regret_epsilon:.2f}")
        print(f"Thompson Sampling Cumulative Regret: {cumulative_regret_thompson:.2f}")


#--------------------------------------#

class EpsilonGreedy(Bandit):
    def __init__(self, true_rewards, epsilon=0.1):
        super().__init__(true_rewards)
        self.epsilon = epsilon
        self.true_rewards = true_rewards
        self.action_counts = [0] * len(true_rewards)
        self.action_values = [0.0] * len(true_rewards)  # Initialize estimated action values


    def __repr__(self):
        return f"EpsilonGreedy Bandit with epsilon={self.epsilon}"

    def pull(self):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.true_rewards) - 1)
        else:
            return self.action_values.index(max(self.action_values))
        

    def update(self, arm, reward):
        self.action_counts[arm] += 1
        n = self.action_counts[arm]
        self.action_values[arm] += (1 / n) * (reward - self.action_values[arm])

    def experiment(self, num_trials):
        rewards = []
        for _ in range(num_trials):
            action = self.pull()
            reward = self.true_rewards[action]  # Simulating the reward based on the chosen action
            rewards.append(reward)
            self.update(action, reward)
        return rewards

    def report(self, rewards, name):
        avg_reward = sum(rewards) / len(rewards)
        avg_regret = max(self.true_rewards) - avg_reward
        print(f"{name} Results: Average Reward={avg_reward:.2f}, Average Regret={avg_regret:.2f}")

#--------------------------------------#

class ThompsonSampling(Bandit):
    def __init__(self, p):
        super().__init__(p)
        self.alpha = [1] * len(p)
        self.beta = [1] * len(p)

    def __repr__(self):
        return "ThompsonSampling Bandit"

    def pull(self):
        sampled_means = [random.betavariate(self.alpha[i], self.beta[i]) for i in range(len(self.true_means))]
        return sampled_means.index(max(sampled_means))

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def experiment(self, num_trials):
        rewards = []
        for _ in range(num_trials):
            arm = self.pull()
            reward = self.true_means[arm]  # Simulating a reward based on the true rewards
            self.update(arm, reward)
            rewards.append(reward)
        return rewards

    def report(self):
        avg_reward = sum(self.alpha) / (sum(self.alpha) + sum(self.beta))
        avg_regret = max(self.true_rewards) - avg_reward
        return f"Thompson Sampling Results: Average Reward={avg_reward:.2f}, Average Regret={avg_regret:.2f}"




def comparison(num_trials):
    # Initialize the bandits
    epsilon = 0.1
    epsilon_greedy_bandit = EpsilonGreedy(Bandit_Reward, epsilon)
    thompson_bandit = ThompsonSampling(Bandit_Reward)  # Change p to Bandit_Reward

    
    epsilon_greedy_rewards = []
    thompson_rewards = []

    for _ in range(num_trials):
        # Pull an arm and get the reward for Epsilon-Greedy bandit
        epsilon_greedy_arm = epsilon_greedy_bandit.pull()
        epsilon_greedy_reward = Bandit_Reward[epsilon_greedy_arm]
        epsilon_greedy_bandit.update(epsilon_greedy_arm, epsilon_greedy_reward)
        epsilon_greedy_rewards.append(epsilon_greedy_reward)
# Pull an arm and get the reward for Thompson Sampling bandit
        thompson_arm = thompson_bandit.pull()
        thompson_reward = Bandit_Reward[thompson_arm]
        thompson_bandit.update(thompson_arm, thompson_reward)
        thompson_rewards.append(thompson_reward)

    # Print Epsilon-Greedy results
    avg_epsilon_greedy_reward = sum(epsilon_greedy_rewards) / num_trials
    epsilon_greedy_regret = max(Bandit_Reward) - avg_epsilon_greedy_reward
    print("Epsilon-Greedy Results:")
    print(f"Average Reward: {avg_epsilon_greedy_reward:.2f}")
    print(f"Average Regret: {epsilon_greedy_regret:.2f}")

    # Print Thompson Sampling results
    avg_thompson_reward = sum(thompson_rewards) / num_trials
    thompson_regret = max(Bandit_Reward) - avg_thompson_reward
    print("\nThompson Sampling Results:")
    print(f"Average Reward: {avg_thompson_reward:.2f}")
    print(f"Average Regret: {thompson_regret:.2f}")
    # Create instances of Bandit classes
    epsilon_greedy_bandit = EpsilonGreedy(Bandit_Reward, epsilon)
    thompson_bandit = ThompsonSampling(Bandit_Reward)

    # Lists to store rewards for each bandit
    epsilon_greedy_rewards = epsilon_greedy_bandit.experiment(num_trials)
    thompson_rewards = thompson_bandit.experiment(num_trials)  # Use thompson_bandit


    # Initialize the Visualization class
    vis = Visualization()

    # Visualize the learning process
    vis.plot1(epsilon_greedy_rewards, thompson_rewards)
    vis.plot2(epsilon_greedy_bandit.experiment(num_trials), thompson_bandit.experiment(num_trials))




    # Store rewards in a CSV file
    vis.store_rewards_to_csv(epsilon_greedy_rewards, thompson_rewards)

    # Report cumulative reward and regret
    vis.report_cumulative_reward_and_regret(epsilon_greedy_rewards, thompson_rewards)


comparison(num_trials)



if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
