# Contrastive UCB

This is a Pytorch implementation for our paper on 

Contrastive UCB: 
Provably Efficient Contrastive Self-Supervised Learning in Online Reinforcement Learning

## Install 
To install the requirements, follow these steps:
```bash
# Install requirements
pip install -r requirements.txt

# Install rlpyt
cd src/rlpyt
pip install -e .
```

## Usage

To run SPR-UCB to train Atari-26 benchmark for 100K frames

```bash
python -m scripts.run --game alien --public
python -m scripts.run --game amidar --public
python -m scripts.run --game assault --public
python -m scripts.run --game asterix --public
python -m scripts.run --game bank_heist --public
python -m scripts.run --game battle_zone --public
python -m scripts.run --game boxing --public
python -m scripts.run --game breakout --public
python -m scripts.run --game chopper_command --public
python -m scripts.run --game crazy_climber  --public
python -m scripts.run --game demon_attack --public
python -m scripts.run --game freeway --public
python -m scripts.run --game frostbite --public
python -m scripts.run --game gopher --public
python -m scripts.run --game hero --public
python -m scripts.run --game jamesbond --public
python -m scripts.run --game kangaroo --public
python -m scripts.run --game krull --public
python -m scripts.run --game kung_fu_master --public
python -m scripts.run --game ms_pacman --public
python -m scripts.run --game pong --public
python -m scripts.run --game private_eye --public
python -m scripts.run --game qbert --public
python -m scripts.run --game road_runner --public
python -m scripts.run --game seaquest --public
python -m scripts.run --game up_n_down --public
```

## Execution

The data for separate runs is stored on disk under the 
result directory with filename 
`rlpyt/data/local/<env-id>-<timestamp>/<seed>/`

- `debug.log` Record the epoch, Q-value, Uncertainty-value, scores.
- `progress.csv` Same data as `debug.log` but with csv format.
- `params.json` The hyper-parameters in training.
- `params.pkl` The saved actor-critic network.

We released the 10-seed results of our method at
this [link](https://www.dropbox.com/s/v9mtxuzwabhswxu/score.zip?dl=0).

## What does each file do? 

    .
    ├── scripts
    │   └── run.py                # The main runner script to launch jobs.
    ├── src                     
    │   ├── agent.py              # Implements the Agent API for action selection 
    │   ├── algos.py              # Distributional RL loss and UCB-based exploration
    │   ├── models.py             # Network architecture and forward passes.
    │   ├── rlpyt_atari_env.py    # Slightly modified Atari env from rlpyt
    │   ├── rlpyt_utils.py        # Utility methods that we use to extend rlpyt's functionality
    │   ├── utils.py              # Command line arguments and helper functions
    |   ├-- rlpyt                 # use rlpyt package  
    │
    └── requirements.txt          # Dependencies
