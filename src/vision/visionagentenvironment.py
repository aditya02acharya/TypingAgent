import gym
import yaml
import logging
from os import path

from abstract.environment import AgentEnv
from src.display.touchscreendevice import TouchScreenDevice

from utilities.utils import distance, visual_distance, EMMA_fixation_time


class VisionAgentEnv(AgentEnv):

    def __init__(self, layout_config):
        self.logger = logging.getLogger(__name__)
        self.config_file = None
        if path.exists(path.join('configs', layout_config)):
            with open(path.join("configs", layout_config), 'r') as file:
                self.config_file = yaml.load(file, Loader=yaml.FullLoader)
                self.logger.info("Device Configurations loaded.")
        else:
            self.logger.error("File doesn't exist: Failed to load %s file under configs folder." % layout_config)

        if self.config_file:
            self.device = TouchScreenDevice(self.config_file['layout_file'])
        else:
            self.device = None
        self.eye_location = None
        self.prev_eye_loc = None
        self.target = None
        self.belief_state = None

        self.observation_space = gym.spaces.Box(low=0.0, high=len(self.device.keys)-1, shape=(1,))
        self.logger.debug("State Space: %s" % repr(self.observation_space))
        self.action_space = gym.spaces.Discrete(self.device.layout.shape[0] * self.device.layout.shape[1])
        self.logger.debug("Action Space: %s" % repr(self.action_space))

    def step(self, action):
        print()

    def reset(self):
        """
        Function to be called on start of a trial. It resets the environment
        and sets the initial belief state.
        :return: current belief state.
        """
        self.logger.debug("Resetting Environment for start of new trial.")
        self.eye_location = self.device.start()
        self.logger.debug("Eye initialised to location: {%d, %d}" % (self.eye_location[0], self.eye_location[1]))
        self.target = self.device.get_random_key()
        self.logger.debug("Target key for the trial set to: {%s}" % self.target)
        self.prev_eye_loc = self.eye_location
        self.set_belief()
        return self.belief_state

    def reward(self, action):
        pass

    def render(self, mode='human'):
        pass

    def set_belief(self):
        self.belief_state = repr(self.target)
        self.logger.debug("current belief state is {%s}" % self.belief_state)

    def calculate_mt(self):
        """
        Calculate total eye movement time.
        :return: (mt_enc, mt_exec , mt_enc_l) : tuple containing (encoding_time, execution_time, left_encoding_time).
        """


