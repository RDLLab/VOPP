import sys
import os
import argparse
import yaml
#import pomdp_py
import importlib
import ray
import torch
import random
import time
import copy
import numpy as np
from utils.run_experiments import run_experiments, run_experiment
from pomdp import POMDP
from parallel_ref_solver import ParallelRefSolver

ray.init(num_gpus=1)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

torch.set_printoptions(precision=10, sci_mode=False)

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "None")

def set_seed(seed):
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.use_deterministic_algorithms(True)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

def parse_args():
    # Parse config file (if provided) and set defaults
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, help='Path to YAML problem config file')
    config_args, remaining_argv = pre_parser.parse_known_args()

    defaults = {}
    if config_args.config:
        with open(config_args.config, 'r') as f:
            defaults = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="Run VOPP on a given POMDP problem",
        parents=[pre_parser]
    )
    
    parser.set_defaults(**defaults)
    parser.add_argument("--device", type=str, default=defaults.get('device', None), 
        help="The device to use. Defaults to cuda, if available"
    )
    parser.add_argument("--seed", type=int, default=defaults.get('seed', None), 
        help="The random seed"
    )
    parser.add_argument("--logdir", type=str, default=defaults.get('logdir', None), 
        help="Log directory"
    )
    parser.add_argument("--logfile_postfix", type=str, default=defaults.get('logfile_postfix'), 
        help="Postfix to add to the log file"
    )
    parser.add_argument("--pomdp_model", type=str, default=defaults.get('pomdp_model'), 
        help="The POMDP model to use (e.g. unc_navigation)"
    )
    parser.add_argument("--discount_factor", type=float, default=defaults.get('discount_factor', 0.95), 
        help="The discount factor"
    )
    parser.add_argument("--n_runs", type=int, default=defaults.get('n_runs', 1), 
        help="Number of runs"
    )
    parser.add_argument("--n_steps", type=int, default=defaults.get('n_steps', 100), 
        help="Maximum unmber of planning steps"
    )
    parser.add_argument("--planning_time_per_step", type=float, default=defaults.get('planning_time_per_step', 1.0), 
        help="Maximum planning time per step"
    )
    parser.add_argument("--max_sampled_episodes", type=int, default=defaults.get('max_sampled_episodes', 0), 
        help="Maximum number of sampled episodes for building the search tree"
    )
    parser.add_argument("--max_search_depth", type=int, default=defaults.get('max_search_depth', 10), 
        help="Maximum depth of the belief tree"
    )    
    parser.add_argument("--num_envs", type=int, default=defaults.get('num_envs', 1), 
        help="Number of parallel simulations for the forward search"
    )
    parser.add_argument("--eta", type=float, default=defaults.get('eta', 0.2), 
        help="Eta parameter"
    )
    parser.add_argument("--alpha_t", type=float, default=defaults.get('alpha_t', 1.0))
    parser.add_argument("--beta_t", type=float, default=defaults.get('beta_t', 1.0))
    parser.add_argument("--num_belief_particles", type=int, default=defaults.get('num_belief_particles', 1000), 
        help="Number of particles used to represent beliefs"
    )
    parser.add_argument('--simulate_headless', action='store_true', default=defaults.get('simulate_headless', False), 
        help="Simulate headless"
    ) 
    args_cli = parser.parse_args(remaining_argv)

    # Inject any keys from the YAML config that aren't already in args_cli
    for k, v in defaults.items():
        if not hasattr(args_cli, k):
            setattr(args_cli, k, v)

    # Postprocessing logic
    if args_cli.device is None:
        args_cli.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args_cli.seed is None:
        args_cli.seed = random.randint(0, 10000)

    if (
        args_cli.max_sampled_episodes > 0 and 
        args_cli.max_sampled_episodes < args_cli.num_envs
    ):
        args_cli.num_envs = args_cli.max_sampled_episodes

    assert (
        args_cli.max_sampled_episodes > 0 or 
        args_cli.planning_time_per_step > 0
    ), "Either 'planning_time_per_step' or 'max_sampled_episodes' must be >0"

    args_cli.headless = args_cli.simulate_headless

    setattr(args_cli, 'file_logging', False)
    if args_cli.logdir is not None:
        args_cli.file_logging = True
    return args_cli

def get_registered_modules(base_path, module_key):
    """Dynamically collect registered models from all submodules of base_path with the given module_key."""
    registered_models = {}   
    
    for module_name in os.listdir(os.path.join(os.path.dirname(__file__), base_path)):
        module_path = f"{base_path}.{module_name}"
        init_file = os.path.join(os.path.dirname(__file__), base_path, module_name, "__init__.py")        
        if os.path.isdir(os.path.join(os.path.dirname(__file__), base_path, module_name)) and os.path.exists(init_file): 
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, module_key):                    
                    registered_models.update(getattr(module, module_key))
            except Exception as e:                
                print(f"Exception when importing module: {e}")           
    
    return registered_models

def setup_problem(args_cli):    
    print(f"seed={args_cli.seed}")    
    set_seed(args_cli.seed)   
    
    # Load registered pomdp_problems
    registered_models = get_registered_modules("pomdp_problems", "REGISTERED_MODELS")
    GenerativeModel = registered_models[args_cli.pomdp_model]    

    # Make generative model for execution
    generative_model_exec = ray.remote(num_gpus=0.01)(GenerativeModel).remote(        
        args_cli=args_cli,
        num_envs=args_cli.num_envs, 
        role='exec',
    )   

    # Make generative model for planning
    generative_model_planning = GenerativeModel(        
        args_cli=copy.deepcopy(args_cli),
        num_envs=args_cli.num_envs,
        role='planning',        
        exec_env=generative_model_exec,
    )    
   
    pomdp_model = POMDP(
        generative_model_planning, 
        num_belief_particles=args_cli.num_belief_particles,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
    )        

    planner = ParallelRefSolver(
        args_cli,
        pomdp_model,            
    )

    return {
        'planner': planner,
        'generative_model_exec': generative_model_exec,        
    }


def main(): 
    args_cli = parse_args()
    with torch.no_grad():
        problem = setup_problem(args_cli)
        run_experiments(
            problem['generative_model_exec'], 
            problem['planner'], 
            runs=args_cli.n_runs, 
            primitive_steps=args_cli.n_steps,
            discount_factor=args_cli.discount_factor, 
            seeds=[args_cli.seed for i in range(args_cli.n_runs)],
            log_directory=args_cli.logdir,
            file_logging=args_cli.file_logging,
            logfile_postfix=args_cli.logfile_postfix
        )

if __name__ == "__main__":
    main()
