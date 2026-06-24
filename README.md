# Vectorized Online POMDP Planner (VOPP)
⚠️ This is an alpha release and the code may contain bugs.

VOPP is a massively parallel online POMDP solver. It represents planning as a sequence of tensor operations and computes policies from tens of thousands of parallel simulations on modern GPUs, without requiring synchronization between concurrent simulations.

This repository contains the official implementation of:

**Vectorized Online POMDP Planning** [[Paper]](https://arxiv.org/abs/2510.27191)

by [Marcus Hoerger](mailto:marcus.hoerger@anu.edu.au), [Muhammad Sudrajat](mailto:muhammad.sudrajat@anu.edu.au), and [Hanna Kurniawati](mailto:hanna.kurniawati@anu.edu.au).

## Citation
If you use this repository in your research, please cite the paper as follows:
```
@inproceedings{hoerger2026VOPP,
  title={Vectorized Online POMDP Planning},
  author={Marcus Hoerger and Muhammad Sudrajat and Hanna Kurniawati},
  booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026},
  url={https://arxiv.org/abs/2510.27191}
}
```

## Requirements
This project requires Python 3.11+ and the following packages:

- PyTorch >= 2.7.1  
- Matplotlib >= 3.10  
- NumPy >= 2.3  
- PyYAML >= 6.0  
- SciPy >= 1.16

## Installation
First, prepare a Conda environment and activate it via
```bash
conda create --name <MY_ENV_NAME> python=3.11
conda activate <MY_ENV_NAME>
```
Next, clone this repository and install its requirements via
```bash
git clone git@github.com/RDLLab/VOPP.git <VOPP_DIR>
pip install -r <VOPP_DIR>/requirements.txt
```
where `<VOPP_DIR>` is a directory of your choice.
		 
## Running VOPP
To run VOPP on the provided benchmark problems, activate your Conda environment and run
```bash
python <VOPP_DIR>/run_vopp.py --config <VOPP_DIR>/configs/<PROBLEM_CONFIGURATION>.yaml
```  
For instance, to solve the Multi-Agent RockSample problem with VOPP, use
```bash
python <VOPP_DIR>/run_vopp.py --config <VOPP_DIR>/configs/ma_rocksample.yaml
```

## Implementing new POMDP problems

### 1. Define a generative model

New POMDP problems can be added by implementing a generative model class that inherits from `pomdp_problems/base/generative_model.py` and provides the following methods detailed in [Generative Model Implementation Details](#4-generative-model-implementation-details):

```python
import torch
from typing import Optional
from pomdp_problems.base.generative_model import GenerativeModel

class MyPOMDPProblem(GenerativeModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ...
                
    @property
    def num_actions(self) -> int:
        ...

    def sample_initial_belief(self, num_samples: int = 1) -> torch.Tensor:
        ...
        
    def sample(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        **kwargs,
    ) -> dict:
        ...

    def likelihood(
        self,
        observation: torch.Tensor,
        prev_state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        log_likelihood: bool = False,
    ) -> torch.Tensor:
        ...

    def heuristics(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        current_nodes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...

```

After implementing the model, place the Python file in a new directory under `pomdp_problems/`, for example:

```text
pomdp_problems/
└── my_pomdp_problem/
    ├── my_pomdp_problem.py
    └── __init__.py

```

The `__init__.py` file should register the new model:

```python
from .my_pomdp_problem import MyPOMDPProblem

REGISTERED_MODELS = {
    "my_pomdp_problem": MyPOMDPProblem,
}

```

The problem can then be referenced from a configuration file using the registered model name, here `"my_pomdp_problem"`.

### 2. Add a problem configuration file 
A problem configuration is specified using a YAML file, where various planner and problem-specific parameters can be configured. An example can be found in `./configs/unc_navigation.yaml`. 

The field `pomdp_model` must be set to the generative model name registered in the problem's `REGISTERED_MODELS` dictionary. For example, if the model was registered as:

```python
REGISTERED_MODELS = {
    "my_pomdp_problem": MyPOMDPProblem,
}
```

then the config file should contain:

```yaml
pomdp_model: my_pomdp_problem
```
Additional problem-specific parameters can be specified directly in the YAML file:
```yaml
my_parameter_a: 0.5
my_parameter_b: True
```
 These parameters can then be accessed inside the generative model, e.g.,
 ```python
 def __init__(self, **kwargs):
     super().__init__(**kwargs)
     args_cli = kwargs.get('args_cli')
     self.my_parameter_a = args_cli.my_parameter_a
     self.my_parameter_b = args_cli.my_parameter_b
 ``` 

### 3. Solve your POMDP Problem with VOPP
Once the generative model and configuration file have been implemented, the problem can be solved using:
```bash
python <VOPP_DIR>/run_vopp.py --config <PATH_TO_CONFIG>.yaml
```

### 4. Generative model implementation details

#### Action and observation representation

Our implementation of VOPP assumes **discrete actions** and **discrete observations**. Actions and observations are represented by integer IDs.

Actions passed to the generative model always have shape:

```python
action.shape == (B,)
```

where each entry is an integer action ID in:

```python
{0, ..., num_actions - 1}
```

The semantics of each action ID are defined by the generative model.

For example, in the `unc_navigation` problem (`pomdp_problems/unc_navigation/unc_navigation.py`), action IDs correspond to motion commands:

```text
0 -> NORTH
1 -> NORTH-EAST
2 -> EAST
...
```

Another example is the MultiAgent-RockSample problem (`pomdp_problems/ma_rocksample/ma_rocksample.py`), where each action ID represents a joint action for both rovers. 


Similarly, observations returned by `sample()` must have shape:

```python
observation.shape == (B,)
```

where each entry is an integer observation ID. The semantics of each observation ID are defined by the generative model.

For example, in the MultiAgent-RockSample problem (`pomdp_problems/ma_rocksample/ma_rocksample.py`), observations consist of one observation per rover. These joint observations are encoded into a single observation ID before being returned by `sample()`.

VOPP only operates on batched action IDs and observation IDs. Any problem-specific action or observation encoding and decoding must be handled inside the generative model.

#### num_actions()
The `num_actions` property must return the number of discrete actions in the problem.

#### sample_initial_belief()

The `sample_initial_belief` method should return `num_samples` particles sampled from the initial belief distribution:

```python
belief_particles.shape == (num_samples, state_dim)
```

#### sample()

The `sample` method defines the generative model. It receives a batch of states and actions:

```python
state.shape  == (B, state_dim)
action.shape == (B,)
```

and should return a dictionary containing the sampled next states, observations, rewards, and terminal flags:

```python
{
    "next_state": next_state,   # (B, state_dim)
    "observation": observation, # (B,)
    "reward": reward,           # (B,)
    "terminal": terminal,       # (B,)
}
```


#### likelihood()

The `likelihood` method is used during belief updates and computes the observation likelihood for a batch of candidate transitions. It receives:

```python
observation.shape == (B,)
prev_state.shape  == (B, state_dim)
next_state.shape  == (B, state_dim)
action.shape      == (B,)
```

and should return the likelihood of the observation for each candidate transition:

```python
likelihood.shape == (B,)
```

If `log_likelihood=False`, the method should return likelihood values. If `log_likelihood=True`, it should return log-likelihood values instead.

#### heuristics()

The `heuristics` method provides a heuristic value estimate used by the planner. It receives:

```python
state.shape  == (B, state_dim)
action.shape == (B,)
```

and should return:

```python
heuristic_values.shape == (B,)
```

## Contact

For questions, issues, or suggestions, feel free to get in touch:

**Marcus Hoerger**  
marcus.hoerger@anu.edu.au

