import time
import random
from collections import OrderedDict
from simulator import Simulator


TRAFIL_LIGHTS_UPDATE_EACH = 5 #periods

class TrafficLight(object):
    """A traffic light that switches periodically."""
    
    valid_states = [True, False]  # True = NS open, False = EW open

    def __init__(self, state=None, period=None):
        self.state = state if state is not None else random.choice(self.valid_states)
        self.period = period if period is not None else random.choice([3, 4, 5])
        self.last_updated = 0

    def reset(self): #called just once
        self.last_updated = 0

    def update(self, t):
        if t - self.last_updated >= self.period:
            self.state = not self.state  # assuming state is boolean
            self.last_updated = t
            
            
class Environment(object):
    """Environment within which all agents operate."""
    valid_actions = [None, 'forward', 'left', 'right']
    valid_inputs = {'light': TrafficLight.valid_states, 'oncoming': valid_actions, 'left': valid_actions, 'right': valid_actions}
    valid_headings = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # ENWS

    start_location = None
    destination = None

    def set_start_location_and_dest(self, start, dest):
        self.start_location = start
        self.destination = dest

    
    def __init__(self, logfilepath=None):
        self.logfilepath = logfilepath
        self.done = False
        self.t = 0
        self.agent_states = OrderedDict() # Stores the dummy agents
        self.status_text = ""

        # Road network
        self.grid_size = (8, 6)  # (cols, rows)
        self.bounds = (1, 1, self.grid_size[0], self.grid_size[1])
        self.block_size = 100
        self.intersections = OrderedDict()
        self.roads = []
		
        for x in xrange(self.bounds[0], self.bounds[2] + 1):
            for y in xrange(self.bounds[1], self.bounds[3] + 1):
                self.intersections[(x, y)] = TrafficLight(period=TRAFIL_LIGHTS_UPDATE_EACH)

        for a in self.intersections:
            for b in self.intersections:
                if a == b:
                    continue
                if (abs(a[0] - b[0]) + abs(a[1] - b[1])) == 1:  # L1 distance = 1
                    self.roads.append((a, b))

        # Dummy agents
        self.num_dummies = 5  # no. of dummy agents
        for i in xrange(self.num_dummies):         
            a = self.create_agent(DummyAgent)
            a.setId(i+1)

        # Primary agent
        self.primary_agent = None  # to be set explicitly
        self.enforce_deadline = False

    def create_agent(self, agent_class, *args, **kwargs):
        agent = agent_class(self, *args, **kwargs)
        self.agent_states[agent] = {'location': random.choice(self.intersections.keys()), 'heading': (0, 1)}
        return agent


    def get_agent_location(self, agent):
        return self.agent_states[agent]['location']

    def set_primary_agent(self, agent, enforce_deadline=False):
        self.primary_agent = agent
        self.enforce_deadline = enforce_deadline

    def reset(self):
        self.done = False
        self.t = 0

        # Reset traffic lights
        for traffic_light in self.intersections.itervalues():
            traffic_light.reset()

        if self.start_location != None and self.destination != None:
            start = self.start_location
            destination = self.destination
        else:
            # Pick a start and a destination
            start = random.choice(self.intersections.keys())
            destination = random.choice(self.intersections.keys())

            # Ensure starting location and destination are not too close
            while self.compute_dist(start, destination) < 4:
                start = random.choice(self.intersections.keys())
                destination = random.choice(self.intersections.keys())




        start_heading = random.choice(self.valid_headings)
        deadline = self.compute_dist(start, destination) * 5
        print "Environment.reset(): Trial set up with start = {}, destination = {}, deadline = {}".format(start, destination, deadline)

        # Initialize agent(s)
        for agent in self.agent_states.iterkeys():
            self.agent_states[agent] = {
                'location': start if agent is self.primary_agent else random.choice(self.intersections.keys()),
                'heading': start_heading if agent is self.primary_agent else random.choice(self.valid_headings),
                'destination': destination if agent is self.primary_agent else None,
                'deadline': deadline if agent is self.primary_agent else None}
            agent.reset(destination=(destination if agent is self.primary_agent else None))

    def step(self):
        print "Environment.step(): t = {}".format(self.t)  # [debug]

        # Update traffic lights
        for intersection, traffic_light in self.intersections.iteritems():
            traffic_light.update(self.t)

        # Update agents
        for agent in self.agent_states.iterkeys():
            agent.update(self.t)

        self.t += 1
        if self.primary_agent is not None:
            if self.enforce_deadline and self.agent_states[self.primary_agent]['deadline'] <= 0:
                self.done = True
                print "Environment.reset(): Primary agent could not reach destination within deadline!"
                self.logEndMessage("Primary agent could not reach destination within deadline!\n")



            self.agent_states[self.primary_agent]['deadline'] -= 1
        print ""

    def sense(self, agent):
        assert agent in self.agent_states, "Unknown agent!"

        state = self.agent_states[agent]    
        location = state['location']
        heading = state['heading']
        light = 'green' if (self.intersections[location].state and heading[1] != 0) or ((not self.intersections[location].state) and heading[0] != 0) else 'red'


        # Populate oncoming, left, right
        oncoming = None
        left = None
        right = None
        for other_agent, other_state in self.agent_states.iteritems():
            if agent == other_agent or location != other_state['location'] or (heading[0] == other_state['heading'][0] and heading[1] == other_state['heading'][1]):
                continue
            other_heading = other_agent.get_next_waypoint()
            if (heading[0] * other_state['heading'][0] + heading[1] * other_state['heading'][1]) == -1:
                if oncoming != 'left':  # we don't want to override oncoming == 'left'
                    oncoming = other_heading
            elif (heading[1] == other_state['heading'][0] and -heading[0] == other_state['heading'][1]):
                if right != 'forward' and right != 'left':  # we don't want to override right == 'forward or 'left'
                    right = other_heading
            else:
                if left != 'forward':  # we don't want to override left == 'forward'
                    left = other_heading

        return {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}  # TODO: make this a namedtuple

    def get_deadline(self, agent):
        return self.agent_states[agent]['deadline'] if agent is self.primary_agent else None

    def logEndMessage(self, message):
        with open(self.logfilepath, 'a') as file:
            file.write(message)


    def act(self, agent, action):
        assert agent in self.agent_states, "Unknown agent!"
        assert action in self.valid_actions, "Invalid action!"

        state = self.agent_states[agent]
        location = state['location']
        heading = state['heading']
        light = 'green' if (self.intersections[location].state and heading[1] != 0) or ((not self.intersections[location].state) and heading[0] != 0) else 'red'


        # Move agent if within bounds and obeys traffic rules
        reward = 0  # reward/penalty
        move_okay = True
        if action == 'forward':
            if light != 'green':
                move_okay = False
        elif action == 'left':
            if light == 'green':
                heading = (heading[1], -heading[0])
            else:
                move_okay = False
        elif action == 'right':
#            if light == 'green' or sense['left'] != 'straight':
#                heading = (-heading[1], heading[0])
#            else:
#                move_okay = False
            #https://discussions.udacity.com/t/are-the-zipped-files-an-old-version-of-the-project/173557
            heading = (-heading[1], heading[0])


        if action is not None:
            if move_okay:
                location = ((location[0] + heading[0] - self.bounds[0]) % (self.bounds[2] - self.bounds[0] + 1) + self.bounds[0],
                            (location[1] + heading[1] - self.bounds[1]) % (self.bounds[3] - self.bounds[1] + 1) + self.bounds[1])  # wrap-around
                #if self.bounds[0] <= location[0] <= self.bounds[2] and self.bounds[1] <= location[1] <= self.bounds[3]:  # bounded
                state['location'] = location
                state['heading'] = heading
                reward = 2 if action == agent.get_next_waypoint() else 0.5
            else:
                reward = -1
        else:
            reward = 1

        if agent is self.primary_agent:
            if state['location'] == state['destination']:
                if state['deadline'] >= 0:
                    reward += 10  # bonus
                self.done = True
                print "Environment.act(): Primary agent has reached destination!\n"
                self.logEndMessage("Primary agent has reached destination!\n")

            self.status_text = "state: {}\naction: {}\nreward: {}".format(agent.get_state(), action, reward)
            #print "Environment.act() [POST]: location: {}, heading: {}, action: {}, reward: {}".format(location, heading, action, reward)  # [debug]

        return reward

    def compute_dist(self, a, b):
        """L1 distance between two points."""
        return abs(b[0] - a[0]) + abs(b[1] - a[1])



class Agent(object):
    """Base class for all agents."""

    def __init__(self, env):
        print "Agent __init__"
        self.env = env
        self.state = None
        self.next_waypoint = None
        self.color = 'cyan'

    def reset(self, destination=None):
        pass

    def update(self, t):
        pass

    def get_state(self):
        return self.state

    def get_next_waypoint(self):
        return self.next_waypoint

    #Function to get the Agent's location
    def get_my_location(self):
        state = self.env.agent_states[self]
        location = state['location']
        heading = state['heading']
        return location, heading

    #Function created to avoid code duplication - also called from Agent.py
    def check_if_action_is_ok(self, inputs):
        action_okay = True

        if self.next_waypoint == 'right':
            if inputs['light'] == 'red' and inputs['left'] == 'forward':
                action_okay = False
        elif self.next_waypoint == 'forward':
            if inputs['light'] == 'red':
                action_okay = False
        elif self.next_waypoint == 'left':
            if inputs['light'] == 'red' or (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
                action_okay = False
        return action_okay

    #This function randomly selects the next action respecting the boundaries of the map
    def get_next_waypoint_given_location(self,loc=None, hed=None):
        topboundary = self.env.bounds[0]
        leftboundary = self.env.bounds[1]

        rightboundary = self.env.bounds[2]
        bottonboundary = self.env.bounds[3]

        options = Environment.valid_actions[1:]
        if loc !=None and hed != None:
            #do not use elses due to the corners
            if loc[0] == leftboundary: #1
                if hed == (0, -1): #N
                    options.remove('left')
                elif hed == (-1, 0): #W
                    options.remove('forward')
                elif hed == (0,1): #S
                    options.remove('right')
            if loc[0] == rightboundary: #8
                if hed == (0, -1): #N
                    options.remove('right')
                elif hed == (1,0): #E
                    options.remove('forward')
                elif hed == (0,1): #S
                    options.remove('left')
            if loc[1] == topboundary: #1
                if hed == (0, -1): #N
                    options.remove('forward')
                elif hed == (1,0): #E
                    options.remove('left')
                elif hed == (-1,0): #S
                    options.remove('right')
            if loc[1] ==bottonboundary:#6
                if hed == (0, 1): #S
                    options.remove('forward')
                elif hed == (1,0): #E
                    options.remove('right')
                elif hed == (-1,0): #W
                    options.remove('left')

        option = random.choice(options)
        return option

class DummyAgent(Agent):
    color_choices = ['blue', 'cyan', 'magenta', 'orange']
    id = 0

    def __str__(self):
        return str(self.id)

    def setId(self, _id):
        self.id = _id

    def __init__(self, env):
        super(DummyAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.next_waypoint = self.get_next_waypoint_given_location()# random.choice(Environment.valid_actions[1:])
        self.color = random.choice(self.color_choices)
        print self.id

    def update(self, t):
        action = None
        inputs = self.env.sense(self)


        action_okay = self.check_if_action_is_ok(inputs)


        if action_okay:
            action = self.next_waypoint
        else:
            loc, hed = self.get_my_location()
            #print "Agent", self, 'Waiting at location:',loc, 'Heading:', hed, 'next_WP:', action

        reward = self.env.act(self, action)
        if action_okay:
            loc, hed = self.get_my_location() #This is the new location, after the agent moved
            self.next_waypoint = self.get_next_waypoint_given_location(loc, hed)





        #print "DummyAgent.update(): t = {}, inputs = {}, action = {}, reward = {}".format(t, inputs, action, reward)  # [debug]
        #print "DummyAgent.update(): next_waypoint = {}".format(self.next_waypoint)  # [debug]

