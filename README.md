# Touchscreen Typing as Optimal Adaptation

This project presents a computational model of how people type on touch-screen keyboards. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Project uses Anaconda for open source python distribution. Goto https://www.anaconda.com/ for installation.
 

The project also uses GPU for training neural network based AI agents. For GPU support follow the link below

```https://docs-cupy.chainer.org/en/stable/install.html#install-cupy```

### Installing

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
To train the agents, use the command below.
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

## Authors

* **Jussi P.P. Jokinen**
* **Aditya Acharya**
* **Mohammad Uzair**
* **Xinhui Jiang**
* **Antti Oulasvirta**

## License


## Acknowledgments
