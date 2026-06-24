
# Vectorized Online POMDP Planner (VOPP)
⚠️ This is an alpha release and the code may contain bugs.

This repository contains the source code for the paper.

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
- Ray >= 2.49  
- SciPy >= 1.16

## Installation
First, prepare a Conda environment and activate it via

	conda create --name <MY_ENV_NAME> python=3.11
	conda activate <MY_ENV_NAME>

Next, clone this repository and install its requirements via

	git clone git@github.com/RDLLab/VOPP.git <VOPP_DIR>
	pip install -r <VOPP_DIR>/requirements.txt

where ``<VOPP_DIR>`` is a directory of your choice.
		 
## Running VOPP
To run VOPP on the provided benchmark problems, activate your Conda environment and run

    python <VOPP_DIR>/run_vopp.py --config <VOPP_DIR>/configs/<PROBLEM_CONFIGURATION>.yaml
  
  For instance, to solve the Multi-Agent Rocksample problem with VOPP, use

	python <VOPP_DIR>/run_vopp.py --config <VOPP_DIR>/configs/ma_rocksample.yaml

