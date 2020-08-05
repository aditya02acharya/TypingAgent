import csv
import tqdm
import logging
import numpy as np
from os import path

import chainer
import chainerrl
import chainer.links as L
from chainerrl import misc
import chainer.functions as F
from chainer import serializers
from chainerrl.agents import PPO
from chainer.backends import cuda
from chainerrl import experiments
from chainer.backends import cuda

from src.abstract.agent import Agent
from src.visualise.visualise import visualise_agent
from src.supervisor.supervisor_agent_environment import SupervisorEnvironment


class SupervisorAgent(Agent):

    def __init__(self, layout_config, agent_params, train):
        self.logger = logging.getLogger(__name__)

        self.env = SupervisorEnvironment(layout_config, agent_params, train)

        optimizer_name = 'Adam' if agent_params is None else agent_params['supervisor']['optimizer_name']
        lr = 0.001 if agent_params is None else agent_params['supervisor']['learning_rate']
        n_units = 512 if agent_params is None else int(agent_params['supervisor']['n_units'])
        device_id = 0 if agent_params is None else int(agent_params['supervisor']['device_id'])
        pre_load = False if agent_params is None else bool(agent_params['supervisor']['pre_load'])
        self.gpu = True if agent_params is None else bool(agent_params['supervisor']['gpu'])
        self.save_path = path.join('data', 'models', 'supervisor') if agent_params is None \
            else agent_params['supervisor']['save_path']
        self.episodes = 1000000 if agent_params is None else int(agent_params['supervisor']['episodes'])
        self.log_interval = 1000 if agent_params is None else int(agent_params['supervisor']['log_interval'])
        self.log_filename = agent_params['supervisor']['log_file']

        winit_last = chainer.initializers.LeCunNormal(1e-2)

        self.model = chainer.Sequential(
            L.Linear(None, n_units),
            F.relu,
            L.Linear(None, n_units),
            F.relu,
            chainerrl.links.Branched(
                chainer.Sequential(
                    L.Linear(None, self.env.action_space.n, initialW=winit_last),
                    chainerrl.distribution.SoftmaxDistribution,
                ),
                L.Linear(None, 1)
            )
        )

        if pre_load:
            serializers.load_npz(path.join(self.save_path, 'best', 'model.npz'), self.model)

        if self.gpu:
            self.model.to_gpu(device_id)

        if optimizer_name == 'Adam':
            self.optimizer = chainer.optimizers.Adam(alpha=lr)
        elif optimizer_name == 'RMSprop':
            self.optimizer = chainer.optimizers.RMSprop(lr=lr)
        else:
            self.optimizer = chainer.optimizers.MomentumSGD(lr=lr)

        self.optimizer.setup(self.model)

        self.optimizer.add_hook(chainer.optimizer.GradientClipping(1.0))

        phi = lambda x: x.astype(np.float32, copy=False)

        self.agent = PPO(
            self.model,
            self.optimizer,
            phi=phi,
            update_interval=1000,
            standardize_advantages=True,
            entropy_coef=1e-2,
            recurrent=False,
        )

        if train:
            chainer.config.train = True
            # self.pbar = tqdm.tqdm(total=self.episodes)
        else:
            chainer.config.train = False
            self.agent.act_deterministically = False

    def train(self, episodes):
        """
        Trains the model for given number of episodes.
        """

        # progress_bar = ProgressBar(self.pbar, episodes)

        experiments.train_agent_with_evaluation(
            self.agent, self.env,
            steps=episodes,  # Train the agent for 2000 steps
            eval_n_steps=None,  # We evaluate for episodes, not time
            eval_n_episodes=10,  # 10 episodes are sampled for each evaluation
            train_max_episode_len=100,  # Maximum length of each episode
            eval_interval=self.log_interval,  # Evaluate the agent after every 1000 steps
            step_hooks=[],  # add hooks
            logger=self.logger,
            outdir=self.save_path)  # Save everything to 'supervisor' directory

    def evaluate(self, sentence, **kwargs):
        """
        Function to evaluate trained agent.
        :param sentence: sentence to type.
        """

        done = False
        if not (sentence == "" or sentence is None):
            self.env.sentences = [sentence]
        state = self.env.reset()
        while not done:
            action = self.agent.act(state)
            state, reward, done, info = self.env.step(action)
            print(done)

        with open(path.join("data", "output", "SupervisorAgent_vision_test.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.env.eye_test_data)

        with open(path.join("data", "output", "SupervisorAgent_finger_test.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.env.finger_test_data)

        with open(path.join("data", "output", "SupervisorAgent_sentence_test.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.env.sentence_test_data)

        # TODO: This is from legacy code. Need to update.
        visualise_agent(True, True, path.join("data", "output", "SupervisorAgent_vision_test.csv"),
                        path.join("data", "output", "SupervisorAgent_finger_test.csv"),
                        path.join("data", "output", "SupervisorAgent.mp4"))


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
