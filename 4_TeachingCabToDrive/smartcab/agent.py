import os
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd
from tabulate import tabulate

script_dir = os.path.dirname(__file__)
path = "C:/git/UdacityMachineLearningEngineerNanodegree/4_TeachingCabToDrive/smartcab/runreport/output_qlearning.txt"#
#path = os.path.join(script_dir, 'runreport/output_qlearning.txt')


class QTable(object):
    def __init__(self):
        self.Q = dict()

    def get(self, state, action):
        key = (state, action)
        return self.Q.get(key, None)

    def set(self, state, action, q):
        key = (state, action)
        self.Q[key] = q
    
    def prettyPrint(self):
        df = pd.DataFrame(columns=['state','action','reward'])
        for k, v in self.Q.items():
            df.loc[len(df)] = [k[0],k[1],v]
        print 'QTable:'
        print tabulate(df, list(df.columns), tablefmt="grid")

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
        self.QTable = QTable()       # Q(s, a)
        self.pRandomMove = pRandomMove
        self.learning_rate = learning_rate
        self.gamma = gamma      # memory / discount factor of max Q(s',a')

        self.possible_actions = Environment.valid_actions
        with open(path, 'a') as file:
            file.write("\n*** parameters: pRandomMove: {}, Learning Rate: {}, gamma: {}\n".format(self.pRandomMove, self.learning_rate, self.gamma))
            file.write("************************************************\n")


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

    e = Environment(logfilepath=path)  # create environment (also adds some dummy traffic)
    a = e.create_agent(QLearningAgent)
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1)  # reduce update_delay to speed up simulation
    sim.run(n_trials=1)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()