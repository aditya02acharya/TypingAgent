import yaml
import numpy
import logging
from os import path
from datetime import datetime

from utilities.logging_config_manager import setup_logging

from src.vision.visionagentenvironment import VisionAgentEnv

# Initialise random seed.
numpy.random.seed(datetime.now().microsecond)

# Setup Logger.
setup_logging(default_path=path.join("configs", "logging.yml"))
logger = logging.getLogger(__name__)
logger.info("logger is set.")


# load app config.
if path.exists(path.join("configs", "config.yml")):
    with open(path.join("configs", "config.yml"), 'r') as file:
        config_file = yaml.load(file, Loader=yaml.FullLoader)
        logger.info("App Configurations loaded.")
else:
    logger.error("File doesn't exist: Failed to load config.yml file under configs folder.")

vision_agent = VisionAgentEnv(config_file['device_config'])
vision_agent.reset()