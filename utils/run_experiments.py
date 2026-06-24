import logging
import scipy.stats as st
import numpy as np
import time
import torch
import copy
import os
from datetime import datetime
from .logger import LogHelper

def get_mean_std_ci(data, confidence=0.95):
    mean = np.mean(data)
    std = np.std(data)
    #ci = st.t.interval(alpha=0.95, df=len(data) - 1, loc=np.mean(data), scale=st.sem(data))
    ci = st.t.interval(confidence=0.95, df=len(data) - 1, loc=np.mean(data), scale=st.sem(data))

    return mean, std, ci

def get_sum(histogram):
    sum = 0.0
    for x, v in histogram.items():
        sum += v
    return sum

def run_experiments(problem, planner, runs=1, primitive_steps=100, **kwargs):
    """Runs multiple runs on a given problem and summarises results"""

    rewards = {}
    rewards_discounted = {}
    steps_dict = {}
    planning_times = {}
    num_sims = {}
    success_list = []
    failed_list = []
    errors_list = []
    runtime = {}
    error_dict = {}
    
    exp_time = str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

    
    file_logging = kwargs.get("file_logging", False)
    log_directory = kwargs.get("log_directory", None)
    prm_logging = kwargs.get("prm_logging", False)
    env_evolution_steps = kwargs.get("env_evolution_steps", [])
    seeds = kwargs.get("seeds", list(range(runs)))

    if prm_logging:
        prm_logger_name = f"{exp_time}_prm"
        prm_logger = LogHelper.get_logger(prm_logger_name, file_logging, prm_logger_name, log_directory)
        g = problem.agent.policy_model._prm.prm
        prm_logger.info(f"PRM Edges: {g.edges}")
        prm_logger.info(f"PRM Nodes: {g.nodes}")
    
    logfile_postfix = kwargs.get('logfile_postfix', "")

    if logfile_postfix is not None:
        exp_logger_name = f"{exp_time}_{logfile_postfix}_summary"
    else:
        exp_logger_name = f"{exp_time}_summary"
    exp_logger = LogHelper.get_logger(exp_logger_name, file_logging, exp_logger_name, log_directory)

    exp_logger.info("=======================================")
    exp_logger.info(f"Starting Test Runner")
    exp_logger.info(f"Start Time: {exp_time}")
    exp_logger.info(f"Problem: {problem}")    
    exp_logger.info(f"Planner: {planner}")

    for run in range(runs):
        exp_logger.info(f"Starting Run {run + 1}")
        if logfile_postfix is not None:
            logfile_str = f"{exp_time}_{logfile_postfix}_run_{run + 1}"
        else:
            logfile_str = f"{exp_time}_run_{run + 1}"
        print("log_directory", log_directory)
        extras_logdir = None
        if log_directory is not None:
            extras_logdir = os.path.join(log_directory, f"{logfile_str}_extras")        
            os.makedirs(extras_logdir, exist_ok=True)
        LogHelper.setup_base_logger(file_logging, log_directory, logfile_str)

        run_start_time = time.time()
        
        if len(env_evolution_steps):
            problem.reset_environment()

        try:            
            total_reward, total_reward_discounted, total_steps, success, mean_planning_time, mean_num_sims \
                = run_experiment(problem, planner, primitive_steps, seeds[run], extras_logdir, **kwargs)

            rewards[run + 1] = total_reward
            rewards_discounted[run + 1] = total_reward_discounted
            steps_dict[run + 1] = total_steps
            planning_times[run + 1] = mean_planning_time
            num_sims[run + 1] = mean_num_sims
            if success:
                success_list.append(run + 1)
            else:
                failed_list.append(run + 1)
            runtime[run + 1] = time.time() - run_start_time
            exp_logger.info(f"Run {run + 1} is done)\n"
                            f"Total Discounted Reward: {total_reward_discounted} | Total Reward: {total_reward_discounted} | Success: {success}")
            exp_logger.info(f"=== RUN COMPLETE! ===")

        except Exception as ex:
            logging.exception(f"Exception in run {run + 1}")
            errors_list.append(run + 1)
            error_dict[run + 1] = ex
            if runs == 1:
                raise ex

        """
        Wait some time to finish logging, so logs don't overlap.
        """
        time.sleep(3)

    # Clear log handlers in main logger
    LogHelper.clear_log_handlers()

    exp_logger.info("=======================================")

    if runs < 30:
        exp_logger.info("WARNING: Sample size is less than 30. CI approximation may be inaccurate.")

    if runs > 1:
        exp_logger.info("---------Experiment Statistics---------")
        exp_logger.info(f"Number of Runs        : {runs}")
        exp_logger.info("-----------------------")

        # Exit if there's only one successful run
        if len(rewards) < 2:
            exp_logger.info(f"Only 1 run is successful")
            return

        reward_mean, reward_std, reward_ci = get_mean_std_ci(list(rewards.values()))
        reward_d_mean, reward_d_std, reward_d_ci = get_mean_std_ci(list(rewards_discounted.values()))
        steps_mean = np.mean(list(steps_dict.values()))
        success_ratio_all = len(success_list) / runs
        if len(errors_list) == runs:
            completed_runs = 0
            success_ratio_comp = 0.0
        else:
            completed_runs = (runs - len(errors_list))
            success_ratio_comp = len(success_list) / completed_runs
        runtime_mean = np.mean(list(runtime.values()))
        runtime_total = np.sum(list(runtime.values()))

        exp_logger.info(f"Reward Mean           : {reward_mean:.3f}")
        exp_logger.info(f"Reward SD             : {reward_std:.3f}")
        exp_logger.info(f"Reward CI             : {reward_ci}")
        exp_logger.info(f"Reward All            : {rewards}")
        exp_logger.info("-----------------------")
        exp_logger.info(f"Reward Mean (Disc.)   : {reward_d_mean:.3f}")
        exp_logger.info(f"Reward SD (Disc.)     : {reward_d_std:.3f}")
        exp_logger.info(f"Reward CI (Disc.)     : {reward_d_ci}")
        exp_logger.info(f"Reward All (Disc.)    : {rewards_discounted}")
        exp_logger.info("-----------------------")
        exp_logger.info(f"Mean Planning Times   : {planning_times}")
        exp_logger.info(f"Mean Simulations      : {num_sims}")
        exp_logger.info("-----------------------")
        exp_logger.info(f"Steps Mean            : {steps_mean:.2f}")
        exp_logger.info(f"Steps All             : {steps_dict}")
        exp_logger.info("-----------------------")
        exp_logger.info(f"Runtime Mean (sec)    : {runtime_mean:.4f}")
        exp_logger.info(f"Runtime Total (sec)   : {runtime_total:.4f}")
        exp_logger.info(f"Runtime All (sec)     : {runtime}")
        exp_logger.info("-----------------------")
        exp_logger.info(f"Completed Runs        : {completed_runs}")
        exp_logger.info(f"Success Count         : {len(success_list)}")
        exp_logger.info(f"Success Runs          : {success_list}")
        exp_logger.info(f"Failed Runs           : {failed_list}")
        exp_logger.info(f"Success Rate (All)    : {100*success_ratio_all:.2f} %")
        exp_logger.info(f"Success Rate (Compl.) : {100*success_ratio_comp:.2f} %")
        if len(error_dict) > 0:
            exp_logger.info("-----------------------")
            exp_logger.info(f"Error Count           : {len(error_dict)}")
            exp_logger.info(f"Error Runs            : {errors_list}")
            exp_logger.info(f"Errors                : {error_dict}")
        exp_logger.info("-----------------------")

def run_experiment(env_exec, planner, primitive_steps=100, seed=0, extras_logdir=None, **kwargs):
    """Runs and the action-feedback loop for the POMDP and records results.""" 

    reset_done = env_exec.reset()
    logging.info(f"DEVICE: {planner.pomdp_model._device}")
    print("reset_done", reset_done)
    planner.reset()
    total_reward = 0.0
    total_reward_discounted = 0.0
    total_steps = 0
    success = 0
    planning_times_per_step = []
    num_sims_per_step = []

    env_evolution_steps = kwargs.get("env_evolution_steps", [])

    current_state = env_exec.sample_initial_belief(num_samples=1)
    print(f"initial state: {planner.pomdp_model._generative_model.state_repr(current_state)}")

    i = 1
    env_index = 0

    # Generate dummy plan
    planner.plan()
    print("Dummy plan created")
    while total_steps <= primitive_steps:
        planning_start = time.time()

        # Plan action        
        action, planning_statistics = planner.plan()
        elapsed_planning = time.time() - planning_start
        planning_times_per_step.append(elapsed_planning)
        num_sims_per_step.append(planning_statistics['num_sims'])        
        logging.info(f"\n\n===== STEP {i} =====")
        logging.info(planning_statistics['planning_stats_str'])   
        logging.info(planning_statistics['policy'].to_str())
        logging.info(f"\nAction:              {planner.pomdp_model._generative_model.action_repr(action)}")

        # Execute action in the environment                
        output = env_exec.sample(current_state, action)
        current_state = output['next_state']
        observation = output['observation']
        reward = output['reward'].cpu().item()
        terminal = output['terminal'].cpu().item()
        info = output.get("info", None)        
        logging.info(f"Next State:          {planner.pomdp_model._generative_model.state_repr(current_state)}")
        logging.info(f"Observation:         {observation}")
        logging.info(f"Total Steps:         {total_steps}")
        logging.info(f"Reward:              {reward}") 
        logging.info(f"Terminal:            {terminal}")
        if info is not None:
            logging.info(f"Info:            {info}")

        # Update belief      
        if not terminal:
            start_t = time.time()
            planner.pomdp_model.update_belief(action, observation)
            logging.info(f'update_belief took {time.time() - start_t:.5f} seconds.')

        # Plot environment
        planner.pomdp_model._generative_model.plot(
            current_state, 
            action, 
            observation, 
            planner.pomdp_model.belief_particles
        )          

        # Logging
        total_reward += reward
        total_reward_discounted += reward * (kwargs.get('discount_factor') ** i)
        total_steps += 1

        logging.info(f"Cum. Disc. Reward:   {total_reward_discounted}")        
        logging.info(f"-------------------")
        logging.info(f"Planner (Num Sims):  {planning_statistics['num_sims']}")
        logging.info(f"Planner (Time):      {elapsed_planning}")

        logging.info(f"=== STEP COMPLETE ===")

        ######################################################
        if extras_logdir is not None:
            step_dict = {
                'state': current_state.cpu(),
                'belief': planner.pomdp_model.belief_particles[:1000].cpu()
            }   

            extras_log = extras_logdir + f"/step_{i}.pt"
            torch.save(step_dict, extras_log)  
        ###################################################### 

        i += 1

        exec_info = env_exec.get_info()
        if exec_info != "":
            logging.info("=========== EXEC INFO ============")
            logging.info(exec_info)

        # Early exit if terminal              
        if terminal:
            logging.info("Reached a terminal state")
            if planner.pomdp_model.is_goal(current_state).cpu().item():
                success = 1

            break

    mean_planning_time = np.mean(planning_times_per_step)
    mean_num_sims = np.mean(num_sims_per_step)

    return total_reward, total_reward_discounted, total_steps, success, mean_planning_time, mean_num_sims
