import logging
import sys
from itertools import product
from operator import itemgetter
from os import path

import gym
import numpy as np
import pandas as pd
import tqdm
import yaml

from src.abstract.environment import AgentEnv
from src.display.touchscreendevice import TouchScreenDevice
from src.utilities.utils import parse_transition_index, distance, WHo_mt


class FingerAgentEnv(AgentEnv):
    def __init__(self, layout_config, agent_params, train):
        self.logger = logging.getLogger(__name__)
        self.config_file = None
        if path.exists(path.join('configs', layout_config)):
            with open(path.join("configs", layout_config), 'r') as file:
                self.config_file = yaml.load(file, Loader=yaml.FullLoader)
                self.logger.info("Device Configurations loaded.")
        else:
            self.logger.error("File doesn't exist: Failed to load %s file under configs folder." % layout_config)
            sys.exit(0)

        if self.config_file:
            self.device = TouchScreenDevice(self.config_file['layout_file'], self.config_file['config'])
            self.user_distance = self.device.device_params['user_distance']
        else:
            self.device = None

        if train:
            self.random_sample = True
        else:
            self.random_sample = False

        self.finger_location = None
        self.prev_finger_loc = None
        self.finger_loc_entropy = None
        self.max_finger_loc = None
        self.belief_state = None
        self.target = None
        self.belief_state = None
        self.action_type = None
        self.vision_status = None
        self.sat_desired_list = agent_params['sat_desired']
        self.sat_desired = None
        self.sat_true_list = agent_params['sat_true']
        self.sat_true = None
        self.actions = agent_params['action_type']
        self.task_reward = agent_params['reward']
        self.init_entropy = None
        self.model_time = 0.0
        self.transition_file = agent_params['transition']
        self.transition_sample = agent_params['transition_samples']
        self.transition_model = None
        self.n_finger_locations = (self.device.layout.shape[0] * self.device.layout.shape[1])
        self.finger_loc_prob = None

        self.discrete_actions = list(product(list(range(-(self.device.layout.shape[0] - 1),
                                                        self.device.layout.shape[0] - 1, 1)),
                                             list(range(-(self.device.layout.shape[1] - 1),
                                                        self.device.layout.shape[1], 1))))

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=(len(self.device.keys) +  # one-hot encoding of target
                                                       self.n_finger_locations +  # one-hot encoding of finger location
                                                       len(self.sat_desired_list) +  # one-hot encoding of sat desired
                                                       len(self.sat_true_list) +  # one-hot encoding of sat true value
                                                       1 +  # action type.
                                                       1,  # entropy.
                                                       )
                                                )
        self.logger.debug("State Space: %s" % repr(self.observation_space))
        self.action_space = gym.spaces.Discrete(len(self.discrete_actions))
        self.logger.debug("Action Space: %s" % repr(self.action_space))

        self.logger.debug("Initialise Transition model.")
        self.initialise_transition()

    def update_model_time(self, delta):
        """
        Function to update model runtime.
        :param delta: time to increment in sec.
        """
        self.model_time += delta

    def initialise_transition(self):
        """
        Setup function for Transition model.
        """
        self.logger.debug("Checking if initialisation of transition model is needed.")
        if path.exists(path.join('data', 'models', self.transition_file)):
            self.logger.info("Transition model exists. Loading existing model.")
            self.transition_model = pd.read_csv(path.join('data', 'models', self.transition_file), index_col=0)
            if not self.check_transition_model_consistency():
                self.logger.debug("The existing Transition model is not consistent with the current model parameters."
                                  " Creating new model.")
                self.create_transition_model()
            else:
                self.logger.debug("Transition model is consistent. Moving on.")
        else:
            self.logger.info("Transition model doesn't exist. Creating new model.")
            self.create_transition_model()

    def create_transition_model(self):
        """
        Function to create a normalised frequency table for transition model.
        Model captures transition probability of finger from current
        location to next.
        """
        rows = list(product(list(range(self.n_finger_locations)), self.sat_true_list,
                            list(range(len(self.discrete_actions)))))

        cols = list(range(self.n_finger_locations))

        self.transition_model = pd.DataFrame(columns=cols, dtype=np.float32)

        for i in tqdm.tqdm(range(len(rows))):
            # create empty row.
            self.transition_model = self.transition_model.append(
                pd.Series(
                    [0] * len(cols),
                    index=self.transition_model.columns,
                    name=str(rows[i]),
                )
            )

            for s in range(self.transition_sample):
                self.finger_location = self.device.convert_to_ij(rows[i][0])
                new_loc, _ = self.move_finger(rows[i][2], rows[i][1])
                index = new_loc[0] * self.device.layout.shape[1] + new_loc[1]
                self.transition_model.loc[str(rows[i]), index] += 1

        # normalise data.
        self.transition_model = self.transition_model.div(self.transition_model.sum(axis=1), axis=0)

        self.transition_model.to_csv(path.join('data', 'models', self.transition_file))
        self.logger.info("Saved the transition table at {%s}" % path.join('data', 'models', self.transition_file))

    def check_transition_model_consistency(self):
        """
        Function to check if the current parameters match the transition table parameters.
        It checks for finger position space, action space and sat space consistent.
        :return: True if consistent.
        """
        # set of hacks to extract sat and action set.
        index_list = list(map(parse_transition_index, self.transition_model.index.values))
        sat_params = np.unique(np.array(list(map(itemgetter(1), index_list))))
        action_params = np.unique(np.array(list(map(itemgetter(2), index_list))))

        # First check the finger space.
        if not (len(self.transition_model.columns) == self.n_finger_locations):
            self.logger.debug('Finger space mismatch found transition model space is {%d}, current: {%d} ' %
                              (len(self.transition_model.columns), self.n_finger_locations))
            return False
        # Second check for SAT space
        elif not (len(set(self.sat_true_list).intersection(sat_params)) == len(self.sat_true_list)):
            self.logger.debug('SAT space mismatch found transition model space is {%s}, current: {%s} ' %
                              (str(sat_params), str(self.sat_true_list)))
            return False
        # Third check for action space.
        elif not (len(action_params) == len(self.discrete_actions)):
            self.logger.debug('Action space mismatch found transition model space is {%d}, current: {%d} ' %
                              (len(action_params), len(self.discrete_actions)))
            return False
        else:
            return True

    def init_finger_location_prob(self):
        """
        Function to initialise belief distribution of finger position.
        """
        self.finger_loc_prob = np.ones((self.n_finger_locations,)) * (1.0 / self.n_finger_locations)
        self.logger.debug("initialised finger position belief to %s" % str(self.finger_loc_prob))

    def step(self, action):
        """
        Function to perform the action selected by the agent.
        :param action: int value for eye movement action taken by agent.
        :return: tuple <next state, reward, done, info>
        """
        self.logger.debug("Current finger position: {%s}" % str(self.finger_location))
        self.logger.debug("Taking {%s} move with (x.diff, y.diff) : {%s},  sat {%.2f} and desired sat {%.2f}" %
                          (self.actions[self.action_type], str(self.discrete_actions[action]),
                           self.sat_true_list[self.sat_true], self.sat_desired_list[self.sat_desired]))

        # take action.
        _, movement_time = self.move_finger(action, self.sat_desired_list[self.sat_desired])

        # for acting what was the reward.
        reward = self.reward(action, movement_time)

        # Update the finger location probability with the new location after movement.
        self.update_finger_loc_prob(action)

        # update belief state.
        self.set_belief()

        # is action terminal.
        # (for current implementation every action is terminal action).
        done = True

        # currently sending empty dict as info. Can extend it to add something in future.
        return self.belief_state, reward, done, {}

    def reset(self):
        """
        Function to be called on start of a trial. It resets the environment
        and sets the initial belief state.
        :return: current belief state.
        """
        self.logger.debug("Resetting Environment for start of new trial.")
        self.finger_location = self.device.start()
        self.logger.debug("Finger initialised to location: {%d, %d}" %
                          (self.finger_location[0], self.finger_location[1]))
        self.target = self.device.get_random_key()
        self.logger.debug("Target key for the trial set to: {%s}" % self.target)
        self.max_finger_loc = self.finger_location[0] * self.device.layout.shape[1] + self.finger_location[1]
        self.logger.debug("Max Finger location initialised to location: {%d}" % self.max_finger_loc)
        self.model_time = 0.0
        self.action_type = self.actions.index(np.random.choice(self.actions))
        self.vision_status = np.random.choice([True, False])
        self.init_finger_location_prob()
        self.calc_finger_loc_entropy()
        self.init_entropy = self.finger_loc_entropy
        self.sat_desired = self.sat_desired_list.index(np.random.choice(self.sat_desired_list))
        self.sat_true = self.sat_true_list.index(np.random.choice(self.sat_true_list))
        self.set_belief()
        return self.preprocess_belief()

    def reward(self, action, movement_time):
        """
        Function for calculating R(a) = 𝜎 · ℎ · p − 𝑚𝑡, where ℎ=1 if the finger model
        presses the requested target, otherwise ℎ=0 and p=1 if the finger model took peck action,
        otherwise p=0.
        :param action: int value for eye movement action taken by agent.
        :param movement_time: movement time in seconds for taking action.
        :return: reward: float value to denote goodness of action.
        """
        peck_action = 1 if self.action_type == 1 else 0

        hit = 1 if self.is_target() else 0

        reward = (self.task_reward * peck_action * self.sat_desired_list[self.sat_desired] * hit) - movement_time

        self.logger.debug("reward for taking action %d is %.2f" % (action, reward))

        return reward

    def render(self, mode='human'):
        """
        Visualise agent-environment interaction. Currently not implemented.
        We have external python code to do this.
        :param mode: string value to denote human viewable mode or agent training mode.
        """
        pass

    def move_finger(self, action, sigma):
        """
        function to make finger movement. Uses Who model to calculate movement time.
        :param sigma: variance parameter for Who model.
        :param action: int value for eye movement action taken by agent.
        :return: new finger location and movement time in seconds.
        """
        # calculate the movement time.
        delta_coord = self.discrete_actions[action]
        new_finger_loc = [self.finger_location[0] + delta_coord[0], self.finger_location[1] + delta_coord[1]]
        dist = distance(self.finger_location, new_finger_loc)
        dist = self.device.convert_to_meters(dist)
        self.logger.debug("Distance (in meters): %.2f" % dist)
        movement_time = WHo_mt(dist, sigma)
        self.update_model_time(movement_time)
        self.logger.debug("took %f seconds to move eyes." % movement_time)
        new_loc = self.update_sensor_position(action, sigma)

        return new_loc, movement_time

    def update_sensor_position(self, action, sigma):
        """
        Function to update the current sensor location.
        :param sigma: variance parameter for Who model.
        :param action: int value for eye movement action taken by agent.
        :return: new_loc: new row, column of the finger position.
        """
        s1 = np.random.normal(0, sigma)
        s2 = np.random.normal(0, sigma)

        # calculating error blocks
        if s1 > 0.5 or s1 < -0.5:
            e1 = int(s1 / 0.5)
        else:
            e1 = 0
        if s2 > 0.5 or s2 < -0.5:
            e2 = int(s2 / 0.5)
        else:
            e2 = 0

        delta_coord = self.discrete_actions[action]
        # calculating new location
        new_loc_0 = self.finger_location[0] + delta_coord[0] + e1
        new_loc_1 = self.finger_location[1] + delta_coord[1] + e2

        # checking upper and lower bounds for y = (0,3)
        new_loc_0 = int(np.clip(new_loc_0, 0, self.device.layout.shape[0] - 1))
        # checking upper and lower bounds for x = (0,10)
        new_loc_1 = int(np.clip(new_loc_1, 0, self.device.layout.shape[1] - 1))

        new_loc = (new_loc_0, new_loc_1)
        return new_loc

    def calc_finger_loc_entropy(self, normalize=True):
        """
        Function to calculate entropy for finger position.
        :param normalize: flag to normalise entropy value.
        :return:
        """
        # H(X) = - Σ P(x) log P(x)
        H = - np.sum(self.finger_loc_prob * np.log(self.finger_loc_prob))
        self.logger.debug("Entropy calculated to : {%.2f}" % H)
        if normalize:
            # normalized by dividing it by information length.
            H_ = H / np.log(self.n_finger_locations)
            self.logger.debug("Normalised entropy calculated to : {%.2f}" % H_)
        self.finger_loc_entropy = round(H_, 1)

    def set_belief(self):
        """
        Function to update belief state.
        """
        self.belief_state = [np.where(self.device.keys == self.target[0])[0][0], self.max_finger_loc, self.sat_desired,
                             self.sat_true, self.action_type, self.finger_loc_entropy]
        self.logger.debug("current belief state is {%s}" % str(self.belief_state))

    def preprocess_belief(self):
        """
        Function to preprocess belief state for neural network input.
        :return:
        """
        one_hot_target = np.eye(len(self.device.keys))[self.belief_state[0]]
        one_hot_finger_loc = np.eye(self.n_finger_locations)[self.belief_state[1]]
        one_hot_sat_desired = np.eye(len(self.sat_desired_list))[self.belief_state[2]]
        one_hot_sat_true = np.eye(len(self.sat_true_list))[self.belief_state[3]]
        state = np.concatenate((one_hot_target, one_hot_finger_loc, one_hot_sat_desired, one_hot_sat_true,
                                [self.belief_state[4]], [self.belief_state[5]]))

        self.logger.debug("Pre-processed belief state to one-hot encoded vector of size {%s}" % str(state.shape))

        return state.astype(np.float32)

    def is_target(self):
        """
        Function to check if the key finger presses is the target.
        """
        coord = self.device.get_coordinate(self.target[0])

        if self.finger_location[0] in coord[0] and self.finger_location[1] in coord[1]:
            self.logger.debug("Pressed the correct key.")
            return True
        else:
            self.logger.debug("Pressed key {%s}, but target is {%s}" %
                              (self.device.get_character(self.finger_location[0], self.finger_location[1]),
                               self.target[0]))
            return False

    def update_finger_loc_prob(self, action):
        """
        Updates the probabilities of finger locations using Bayes Rule.
        Bayes Update => b`(s`) = Σs∈S T(s,a,s`)*b(s) / Pr(o|a,b)
        """
        probs = {}
        if not self.vision_status:
            # if eye are not present to track finger position.
            # b`(s`) = Σs∈S T(s,a,s`)*b(s)
            for next_loc in self.finger_loc_prob:
                state = list(product(list(range(self.n_finger_locations)), [self.sat_true_list[self.sat_true]],
                                     [action]))
                state = list(map(str, state))

                print(self.transition_model.loc[state, :])
