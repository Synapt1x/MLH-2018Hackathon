#MLH Hackathon 2018
## Group: Chris Cadonic and Cassandra Aldaba

# Project: Detecting patients becoming immobile when waiting in emergency room waiting areas

## Setup

The working environment is stored in the requirements file *requirements.txt*. Environment can be loaded/updated using pip:

`pip install -r requirements.txt`

An initial jupyter notebook *pt_detect.ipynb* has been prepared in case we want to present our work that way. I've also set up traditional python scripts for running the system. Setup is as:

- `main.py`: runs the main program
- `util.py`: contains code for data loading/saving and utility
- `cnn.py`: contains (will contain) code to implement a neural network model
- `lk.py`: contains (will contain) code to implement optical flow using hierarchical LK

In either system, the *config.yaml* contains hyperparameters for cnn training, should we use them, and also general parameters (e.g., working paths) and potential parameters for other approaches.

## Running

As the project is yet to be implemented, there's much to discuss here. Though, it is assumed that if the jupyter notebook route is taken then running the kernel will run cell by cell. Otherwise, `main.py` will handle running the system.
