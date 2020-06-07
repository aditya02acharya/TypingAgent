import csv
import tqdm
import logging
import numpy as np
from os import path
from collections import deque

import chainer
import chainerrl
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer.backends import cuda

from src.abstract.agent import Agent
from src.finger.model import QFunction
from src.finger.finger_agent_environment import FingerAgentEnv


class FingerAgent(Agent):

    def __init__(self, layout_config, agent_params, train):
        self.logger = logging.getLogger(__name__)

        self.env = FingerAgentEnv(layout_config, agent_params, train)

        optimizer_name = 'Adam' if agent_params is None else agent_params['optimizer_name']
        lr = 0.001 if agent_params is None else agent_params['learning_rate']
        dropout_ratio = 2 if agent_params is None else int(agent_params['dropout_ratio'])
        n_units = 512 if agent_params is None else int(agent_params['n_units'])
        device_id = 0 if agent_params is None else int(agent_params['device_id'])
        pre_load = False if agent_params is None else bool(agent_params['pre_load'])
        gpu = False if agent_params is None else bool(agent_params['gpu'])
        self.save_path = path.join('data', 'models', 'finger') if agent_params is None \
            else agent_params['save_path']
        gamma = 0.99 if agent_params is None else float(agent_params['discount'])
        replay_size = 10 ** 6 if agent_params is None else int(agent_params['replay_buffer'])
        self.episodes = 1000000 if agent_params is None else int(agent_params['episodes'])
        self.log_interval = 1000 if agent_params is None else int(agent_params['log_interval'])

        # Agent Configuration.
        self.q_func = QFunction(obs_size=self.env.observation_space.shape[0], n_actions=self.env.action_space.n,
                                n_hidden_channels=n_units, dropout_ratio=dropout_ratio)

        if pre_load:
            serializers.load_npz(path.join(self.save_path, 'best', 'model.npz'), self.q_func)

        if gpu:
            self.q_func.to_gpu(device_id)

        if optimizer_name == 'Adam':
            self.optimizer = chainer.optimizers.Adam(alpha=lr)
        elif optimizer_name == 'RMSprop':
            self.optimizer = chainer.optimizers.RMSprop(lr=lr)
        else:
            self.optimizer = chainer.optimizers.MomentumSGD(lr=lr)

        self.optimizer.setup(self.q_func)

        # Use epsilon-greedy for exploration/exploitation with linear decay.
        explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1.0, end_epsilon=0.1,
                                                                decay_steps=int(self.episodes / 4),
                                                                random_action_func=self.env.action_space.sample)

        # DQN uses Experience Replay.
        replay_buffer = chainerrl.replay_buffers.prioritized.PrioritizedReplayBuffer(capacity=replay_size)

        phi = lambda x: x.astype(np.float32, copy=False)

        # Now create an agent that will interact with the environment.
        self.agent = chainerrl.agents.DoubleDQN(q_function=self.q_func, optimizer=self.optimizer,
                                                replay_buffer=replay_buffer, gamma=gamma, explorer=explorer,
                                                replay_start_size=50000, update_interval=1, target_update_interval=1000,
                                                target_update_method='soft', phi=phi)

        if train:
            chainer.config.train = True
            self.pbar = tqdm.tqdm(total=self.episodes)
        else:
            chainer.config.train = False

    def train(self, episodes):
        """
        Function to start agent training. Finger agent uses Double DQN with Prioritised Experience Reply.
        :param episodes: number of training trials to run.
        """
        progress_bar = ProgressBar(self.pbar, episodes)

        chainerrl.experiments.train_agent_with_evaluation(
            agent=self.agent,
            env=self.env,
            steps=episodes,  # Train the agent for n steps
            eval_n_steps=None,  # We evaluate for episodes, not time
            eval_n_episodes=10,  # 10 episodes are sampled for each evaluation
            eval_interval=self.log_interval,  # Evaluate the agent after every 1000 steps
            outdir=self.save_path,  # Save everything to 'data/models' directory
            train_max_episode_len=5,  # Maximum length of each episode
            successful_score=4.5,  # Stopping rule
            logger=self.logger,
            step_hooks=[progress_bar]
        )

    def evaluate(self, sentence):
        pass


class ProgressBar(chainerrl.experiments.hooks.StepHook):
    """
    Hook class to update progress bar.
    """
    def __init__(self, pbar, max_length):
        self.pbar = pbar
        self.max = max_length

    def __call__(self, env, agent, step):
        self.pbar.update()
        if self.max <= step:
            self.pbar.close()





