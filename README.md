## Overview
------------

PowerGym is a Gym-like environment for Volt-Var control in power distribution systems.

The Volt-Var control targets minimizing voltage violations, control loss, and power loss under physical networked constraints and device constraints. The networked constraints are maintained by the power distribution system simulator, OpenDSS. The device constraints are usually integer constraints on the actions.

Below is a description of observation and action spaces. {} denotes a finite set and [] denote a continuous interval.

|**Observation Space** | |
| ------------- | ------------- |
| **Variable**| **Range**|
| Bus voltage     | [0.8, 1.2] |
| Capacitor status     | {0, 1} |
| Regulator tap number | {0, ..., 32} |
| State-of-charge (soc) | [0, 1] |
| Discharge power  | [-1, 1]  |

|**Action Space** | |
| ------------- | ------------- |
| **Variable**| **Range**|
| Capacitor status     | {0, 1} |
| Regulator tap number | {0, ..., 32} |
| Discharge power (disc.) | {0, ..., 32} |
| Discharge power (cont.) | [-1, 1]  |

There are two kinds of batteries. Discrete battery has discretized choices on the discharge power (e.g., choose from {0,...,32}) and continuous battery chooses the normalized discharge power from the interval [-1,1]. The user should specify the battery's kind upon calling the environment.

The reward function is a combination of three losses: voltage violation, control error, and power loss. The control error is further decomposed into capacitor's & regulator's switching cost and battery's discharge loss & soc loss. The weights among these losses depends on the circuit system and is listed in the Appendix of our paper. 

The implemented circuit systems are summerized as follows.
| **System**| **# Caps**| **# Regs**| **# Bats**|
| ------------- | ------------- |------- |------- |
| 13Bus     | 2 | 3 | 1 |
| 34Bus | 4 | 6 | 2 |
| 123Bus | 4 | 7 | 4 |
| 8500Node | 10 | 12 | 10 |


## Requirements
------------
- Python 3.8

For the complete installation
```
pip install -r requirements.txt
```

## Usage
------------
### Run options
`random_agent.py` gives a minimal example of PowerGym usage. The option `--mode` can choose various running mode

To run PowerGym in a single episode
 ```
 python random_agent.py
 ```

To run PowerGym for parallel environments
```
python random_agent.py --mode=parallele
```

To run PowerGym for multiple episodes
```
python random_agent.py --mode=episodic
```

To run PowerGym using OpenDSS controllers defined in the circuit files (if any) 
```
python random_agent.py --mode=dss
```

### Environment name options
The option `--env_name` can choose various environments. Below, we take 123Bus as an example.

Run a vanilla environment
```
python random_agent.py --env_name 123Bus
```

Run a scaled environment
```
python random_agent.py --env_name 123Bus_s1.5
```

Run an environment with soc error
```
python random_agent.py --env_name 123Bus_soc
```

Run a scaled environment with soc error
```
python random_agent.py --env_name 123Bus_soc_s1.5
```



## Citation

To cite PowerGym, please cite the following paper:

```
@article{fan2021powergym,
  title={PowerGym: A Reinforcement Learning Environment for Volt-Var Control in Power Distribution Systems},
  author={Fan, Ting-Han and Lee, Xian Yeow and Wang, Yubo},
  journal={arXiv preprint arXiv:2109.03970},
  year={2021}
}
```

## License
This project is licensed under MIT License. See [LICENSE.md](LICENSE.md) for more details.
