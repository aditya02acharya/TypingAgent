import sys
import yaml
import numpy
import logging
import argparse
from os import path
from datetime import datetime

from src.utilities.logging_config_manager import setup_logging

from src.display.touchscreendevice import TouchScreenDevice
from src.vision.vision_agent_environment import VisionAgentEnv

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--all", action="store_true", default=False,
                    help="train all the agents [vision, finger, proofread, supervisor]")
parser.add_argument("--vision", action="store_true", default=False, help="train only the vision agent")
parser.add_argument("--finger", action="store_true", default=False, help="train only the finger agent")
parser.add_argument("--proofread", action="store_true", default=False, help="train only the proofread agent")
parser.add_argument("--supervisor", action="store_true", default=False, help="train only the supervisor agent")
parser.add_argument("--train", action="store_true", default=False, help="run model in train mode")
parser.add_argument("--config", required=True, help="name of the configuration file (REQUIRED)")
parser.add_argument("--seed", type=int, default=datetime.now().microsecond, help="random seed default: current time")

# get user command line arguments.
args = parser.parse_args()

# Initialise random seed.
numpy.random.seed(args.seed)

# Setup Logger.
setup_logging(default_path=path.join("configs", "logging.yml"))
logger = logging.getLogger(__name__)
logger.info("logger is set.")

# load app config.
if path.exists(path.join("configs", args.config)):
    with open(path.join("configs", args.config), 'r') as file:
        config_file = yaml.load(file, Loader=yaml.FullLoader)
        logger.info("App Configurations loaded.")
else:
    logger.error("File doesn't exist: Failed to load config.yml file under configs folder.")
    sys.exit(0)

if args.train:
    if path.exists(path.join("configs", config_file['training_config'])):
        with open(path.join("configs", config_file['training_config']), 'r') as file:
            train_config = yaml.load(file, Loader=yaml.FullLoader)
            logger.info("Training Configurations loaded.")
    else:
        logger.error("File doesn't exist: Failed to load %s file under configs folder." %
                     config_file['training_config'])
        sys.exit(0)

    if args.vision or args.all:
        logger.debug("Initiating Vision Agent Training.")
        vision_agent = VisionAgentEnv(config_file['device_config'], train_config['vision'])
        vision_agent.reset()
        vision_agent.step(8)
