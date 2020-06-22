# Touchscreen Typing as Optimal Adaptation

This project presents a computational model of how people type on touch-screen keyboards. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Project uses Anaconda for open source python distribution. Goto https://www.anaconda.com/ for installation.
 

The project also uses GPU for training neural network based AI agents. For GPU support follow the link below

```https://docs-cupy.chainer.org/en/stable/install.html#install-cupy```

### Installing

```
git clone https://github.com/aditya02acharya/TypingAgent.git
```

Project provides an anaconda environment setup file to install project prerequisites. 
Run the command below in the project directory. 

```
conda env create -f configs/env_setup.yml
```

This command creates the anaconda environment **typing**. Use to command below to activate the environment.
```
conda activate typing
```

### Running the scripts
To train the agents, use the command below. Please check and edit the configuration files before running the scripts.
```
python main.py --train --all --config config.yml
```

To evaluate the trained agent, use the command below.
```
python main.py --all --config config.yml --type "hello world>"
```

To see all available argument, use the command below.
```
python main.py --help
```

### Configuration setting
All project configuration files are kept under configs folder.
* **config.yml**: this is the main configuration file. You can link config files for experiments here.
* **device_config.yml**: this file contains the device configuration. For example, layout configuration, key size, etc. 
* **logging.yml**: centralised project logging configuration. Set logging mode to either `INFO`, `DEBUG`, `WARN`, `CRITICAL`. Logs are stored under logs directory.
* **training_config.yml**: model training configuration for each agent.
* **evaluation_config.yml**: model testing configuration for each agent. 

### Project Storage
* **configs**: contains all configuration files.
* **data**: contains model outputs. Trained agent data kept under `models`. Test/Evaluation data kept under `output`.
* **layouts**: contains keyboard layouts. Layouts are stored as 2d-numpy array.
* **logs**: contains log files.
* **src**: contains project code base. 

## Authors

* **Jussi P.P. Jokinen**
* **Aditya Acharya**
* **Mohammad Uzair**
* **Xinhui Jiang**
* **Antti Oulasvirta**

## Contributors
* **Jussi P.P. Jokinen**
* **Aditya Acharya**
* **Mohammad Uzair**

## License


## Acknowledgments
