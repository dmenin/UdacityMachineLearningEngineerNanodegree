import os
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd
from tabulate import tabulate
import numpy as np

script_dir = os.path.dirname(__file__)
#path = "C:/git/UdacityMachineLearningEngineerNanodegree/4_TeachingCabToDrive/smartcab/runreport/output_qlearning.txt"#
path = os.path.join(script_dir, 'runreport/output_qlearning.txt')

DEBUG = False


class QTable(object):
    def __init__(self):
        self.Q = dict()

    def get(self, state, action):
        key = (state, action)
        #default returns 0 - prevents actions with negative rewards from being chosen when there is an action with 0 reward
        return self.Q.get(key, 0)

    def set(self, state, action, q):
        key = (state, action)
        self.Q[key] = q
    
    def prettyPrint(self, returndf=False):
        df = pd.DataFrame(columns=['State (light / oncoming / right / left)','action','reward'])
        for k, v in self.Q.items():
            df.loc[len(df)] = [k[0],k[1],v]
        print 'QTable:'
        print tabulate(df, list(df.columns), tablefmt="grid")

        if returndf:
            return df

    def simplePrint(self):
        for k, v in self.Q.items():
            print k, v



class QLearn(Agent):
    #QLeran Parameterers Cheat Sheet:
    #1) pRandomMove  (Epsilon):
    #       Probability of doing random move
    #       https://www.udacity.com/course/viewer#!/c-ud728-nd/l-5446820041/m-634899065
    #2) learning_rate (Alpha):
    #3) (Gamma):
    #   http://mnemstudio.org/path-finding-q-learning-tutorial.htm
    #   The Gamma parameter has a range of 0 to 1 (0 <= Gamma > 1).  If Gamma is closer to zero, the agent will tend to consider only immediate rewards.
    #   If Gamma is closer to one, the agent will consider future rewards with greater weight, willing to delay the reward.
    def __init__(self, pRandomMove=.1, learning_rate =.5, gamma=.5):
        self.QTable = QTable()

        self.pRandomMove = pRandomMove      #Epsilon
        self.learning_rate = learning_rate  #Alpha
        self.gamma = gamma         # memory / discount factor of max Q(s',a')

        self.possible_actions = Environment.valid_actions
        with open(path, 'a') as file:
            file.write("\nParameters: pRandomMove: {}, Learning Rate: {}, gamma: {}\n".format(self.pRandomMove, self.learning_rate, self.gamma))
            file.write("-------------------------------------------------------\n")

    def GetNextPossibleBestAction(self, state):
        if random.random() < self.pRandomMove:
            if DEBUG:
                print "RANDOM MOVE"
            action = random.choice(self.possible_actions)
        else:
            pr = []
            for pa in self.possible_actions: #for each possible action
                pr.append(self.QTable.get(state, pa)) #get the possible reward from the QTable


            #Get The Max Reward
            max_reward = max(pr)

            #Gets the ID correspondent to the max reward
            #if there are ties or all the rewards are None, gets a random value among those
            action_idx = random.choice([i for i in range(len(self.possible_actions)) if pr[i] == max_reward])
            action = self.possible_actions[action_idx]
            if DEBUG:
                print 'Ations Rewards(N,F,L,R):', pr
                print 'Index Chosen:', action_idx

        return action


    #Implement:
    #Q(s,a) = Q(s,a) + alpha * [R(s,a) + gamma * argmax(R(s', a')) - Q(s, a)]
    def Learn(self, StateWhereIWas, ActionITook, StateWhereIAm, RewardIGot):
        currentQValue = self.QTable.get(StateWhereIWas, ActionITook)
        alpha = self.learning_rate #Adding to a variable for simplicity of the formula only

        #Get the Possible Future Rewards From The Current State
        pr = []
        for pa in self.possible_actions: #for each possible action
            pr.append(self.QTable.get(StateWhereIAm, pa))

        FutureReward = 0 if max(pr) is None else max(pr)
        
        
        #Q(s,a) =      Q(s,a)   + alpha * [R(s,a)     +    gamma   *  argmax(R(s', a')) - Q(s, a)]
        newQ    = currentQValue + alpha * (RewardIGot + self.gamma *    FutureReward    - currentQValue)

        #Update the QTable with the new value
        self.QTable.set(StateWhereIWas, ActionITook, newQ)
        

class QLearningAgent(Agent):

     def __init__(self, env, pRandomMove, learning_rate, gamma):
        super(QLearningAgent, self).__init__(env)
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.possible_actions= Environment.valid_actions
        self.QLearning = QLearn(pRandomMove, learning_rate, gamma)


     def reset(self, destination=None):
        self.planner.route_to(destination)

    #Implements "Method 1" as described here: https://discussions.udacity.com/t/please-someone-clear-up-a-couple-of-points-to-me/45365
    #1) Sense the environment (see what changes naturally occur in the environment)
    #2) Take an action - get a reward
    # 3) Sense the environment (see what changes the action has on the environment)
    # 4) Update the Q-table
    # 5) Repeat
     def update(self, t):
        if DEBUG:
            self.QLearning.QTable.prettyPrint()
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator

        loc = self.env.sense(self).items()
        self.state = (loc[0][1],loc[1][1],loc[2][1],loc[3][1],self.next_waypoint)

        action = self.QLearning.GetNextPossibleBestAction(self.state)
        reward = self.env.act(self, action)#Execute action and get reward (-1, 0.5,1 or 2)
        if DEBUG:
            print 'Action Chosen:', action, 'reward: ', reward


        #Sense the New location
        newloc = self.env.sense(self).items()
        next_state = (newloc[0][1],newloc[1][1],newloc[2][1],newloc[3][1], self.next_waypoint)

        #Even though the agent just moved, the call bellow is from its perspective BEFORE the move, so:
        #   State, Action: What the agent just did (where it was and what it considered the best action)
        #   Next State: Where it is
        #   Reward: it got for the move
        self.QLearning.Learn(self.state, action, next_state, reward)

        deadline = self.env.get_deadline(self)
        if DEBUG:
            print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, loc, action, reward)  # [debug]


#This is the Ramdom Learning Agent, that only takes Random moves
#Is not really a Learning Agent, I just kept the original name to avoid confusion
class LearningAgent(Agent):
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.next_waypoint = None
        self.total_reward = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.next_waypoint = None

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Select action according to your policy
        loc, hed = self.get_my_location()
        action = self.get_next_waypoint_given_location( loc, hed)
        action_okay = self.check_if_action_is_ok(inputs)
        if not action_okay:
            action = None
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward += reward

        #TODO: Learn policy based on state, action, reward
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    #DUMMY RANDOM AGENT...no file log needed
    #e = Environment()  # create environment (also adds some dummy traffic)
    #e.set_start_location_and_dest((1,1), (8,6)) set the Start Location
    #a = e.create_agent(LearningAgent)  # create agent

    ep = 0.1
    al = 0.2
    ga = 0.2

    #pRandomMove\learning_rate\gamma
    #for ep in np.arange(0.1, 0.25, 0.05):
    #for al in np.arange(0.2, 0.45, 0.05):
    #for ga in np.arange(0.2, 0.45, 0.05):
    e = Environment(logfilepath=path) #create environment (also adds some dummy traffic)
    e.DEBUG = DEBUG
    a = e.create_agent(QLearningAgent, pRandomMove=ep, learning_rate = al, gamma=ga)
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    sim = Simulator(e, update_delay=0.1)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100) # press Esc or close pygame window to quit


    print tabulate(e.dfLog, list(e.dfLog.columns), tablefmt="grid")
    #e.dfLog.to_csv("dfLog-100-trials-tak3.csv")


if __name__ == '__main__':
    run()
