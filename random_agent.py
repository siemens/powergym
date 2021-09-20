# Copyright 2021 Siemens Corporation
# SPDX-License-Identifier: MIT

"""Random agent to probe enviroment
"""
import matplotlib.pyplot as plt
import numpy as np
import imageio
import glob

from powergym.env_register import make_env, remove_parallel_dss

import argparse
import random
import itertools
import sys, os
import multiprocessing as mp

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--env_name', default='13Bus')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                         help='random seed')
    parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                         help='maximum number of steps')
    parser.add_argument('--num_workers', type=int, default=3, metavar='N', 
                         help='number of parallel processes')
    parser.add_argument('--use_plot', type=lambda x: str(x).lower()=='true', default=False)
    parser.add_argument('--do_testing', type=lambda x: str(x).lower()=='true', default=False)
    parser.add_argument('--mode', type=str, default='single',
                        help="running mode, random, parallele, episodic or dss")
    args = parser.parse_args()
    return args

def seeding(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_random_agent(args, load_profile_idx=0, worker_idx=None, use_plot=False, print_step=False):
    """Run a agent that takes arbitrary control actions
    
    Args:
        test_action (str): Test env with fixed or random actions 
        steps (int, optional): Number of steps to run the env
        use_plot (bool, optional): Plot outcome of random agent
    """

    cwd = os.getcwd()
    
    # get environment
    env = make_env(args.env_name, worker_idx=worker_idx)
    env.seed(args.seed + 0 if worker_idx is None else worker_idx)

    if print_step:
        print('This system has {} capacitors, {} regulators and {} batteries'.format(env.cap_num, env.reg_num, env.bat_num))
        print('reg, bat action nums: ', env.reg_act_num, env.bat_act_num)
        print('-'*80)

    # assuming we only train on one load profile for now
    obs = env.reset(load_profile_idx=load_profile_idx)
    
    # RL loop
    if use_plot and not os.path.exists(os.path.join(cwd,'random_agent_plots')):
        os.makedirs(os.path.join(cwd,'random_agent_plots'))

    episode_reward = 0.0
    for i in range(env.horizon):
        action = env.random_action()
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        print(worker_idx, i, reward)

        if print_step:
            print('\nStep:{}\n'.format(i))
            print('Action:', action)
            print('Next Obs: {} R: {} Done: {} Info: {}'.format(obs, reward, done, info))

        if use_plot:
            # node_bound argument to decide to plot max or min node voltage for nodes with more than 1 phase
            fig, _ = env.plot_graph()
            fig.tight_layout(pad=0.1)
            fig.savefig(os.path.join(cwd,'random_agent_plots', 'node_voltage_' + str(i).zfill(4) + '.png'))
            plt.close()
    
    print('load_profile: {}, episode_reward: {}'.format(load_profile_idx, episode_reward))

    if use_plot:
        fig, _ = env.plot_graph(show_voltages=False)
        fig.tight_layout(pad=0.1)
        fig.savefig(os.path.join(cwd, 'random_agent_plots', 'system_layout.pdf'))
    
        # make sure to import imageio and glob if generating gif
        # generate gif
        '''
        images = []
        filenames = sorted(glob.glob(os.path.join(cwd, "random_agent_plots/*.png")))
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave(os.path.join(cwd, 'random_agent_plots/node_voltage.gif'), images, fps=1)
        '''

def run_parallel_random_agent(args):
    workers = []
    for i in range(0, args.num_workers):
        p = mp.Process( target=run_random_agent, args=(args,i, i) )
        p.start()
        workers.append(p)
    for p in workers:
        p.join()

    remove_parallel_dss(args.env_name, args.num_workers)

def random_evaluate(env, profiles, episodes=None):
    returns = np.zeros(len(profiles)) if episodes is None else np.zeros(min(episodes, len(profiles)))
    for i in range(len(returns)):
        pidx = profiles[i] if episodes is None else random.choice(profiles)
        obs = env.reset(load_profile_idx = pidx)
        episode_reward = 0
        episode_steps = 0
        done = False
        while not done:
            action = env.random_action()
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            mask = 1 if episode_steps == env.horizon else float(not done)
            obs = next_obs
        returns[i] = episode_reward
    return returns.mean(), returns.std()

def run_episodic_random_agent(args, worker_idx=None):
    """Run a episodic random agent with train-test split
  
    """
    # output file
    if args.do_testing:
        fout = open("result/{}_random_{}.csv".format(args.env_name, args.seed), 'w')

    # get environment
    env = make_env(args.env_name, worker_idx=worker_idx)
    env.seed(args.seed + 0 if worker_idx is None else worker_idx)

    # get obs, act
    obs_dim = env.observation_space.shape[0]
    CRB_num = ( env.cap_num, env.reg_num, env.bat_num )
    CRB_dim = (2, env.reg_act_num, env.bat_act_num )
    print('NumCap, NumReg, NumBat: {}'.format(CRB_num))
    print('ObsDim, ActDim: {}, {}'.format(obs_dim, sum(CRB_num)))
    print('-'*80)

    # train-test split
    if args.do_testing:
        train_profiles = random.sample(range(env.num_profiles), k=env.num_profiles//2)
        test_profiles = [i for i in range(env.num_profiles) if i not in train_profiles]
    else:
        train_profiles = list(range(env.num_profiles))

    # Training Loop
    total_numsteps = 0
        
    for i_episode in itertools.count(start=1):
        
        episode_reward = 0
        episode_steps = 0
        done = False
        load_profile_idx = random.choice(train_profiles)
        obs = env.reset(load_profile_idx = load_profile_idx)

        while not done:
            action = env.random_action()  # Sample random action
            next_obs, reward, done, info = env.step(action)
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            mask = 1 if episode_steps == env.horizon else float(not done)
            obs = next_obs

        print("episode: {}, profile: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, load_profile_idx, total_numsteps, episode_steps, round(episode_reward, 2)))
        
        total_numsteps+=24

        if total_numsteps >= args.num_steps: break
        
        if args.do_testing and i_episode%5==0:
            avg_reward, std = random_evaluate(env, test_profiles)
            fout.write('{},{},{}\n'.format(total_numsteps, avg_reward, std))
            fout.flush()
            print("----------------------------------------")
            print("Avg., Std. Reward: {}, {}".format(round(avg_reward, 2), round(std, 2)))
            print("----------------------------------------")

def run_dss_agent(args):  
    # get environment
    env = make_env(args.env_name, dss_act=True)
    env.seed(args.seed)
    
    profiles = list(range(env.num_profiles))

    # Training Loop
    total_numsteps = 0
        
    for i_episode in itertools.count(start=1):
        episode_reward = 0
        episode_steps = 0
        done = False
        load_profile_idx = random.choice(profiles)
        obs = env.reset(load_profile_idx = load_profile_idx)

        while not done:
            next_obs, reward, done, info = env.dss_step()  
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            obs = next_obs
            
        print("episode: {}, profile: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, load_profile_idx, total_numsteps, episode_steps, round(episode_reward, 2))) 
        if total_numsteps >= args.num_steps: break

if __name__ == '__main__':
    args = parse_arguments()
    seeding(args.seed)
    if args.mode.lower() ==  'single':
        run_random_agent(args, worker_idx=None, use_plot=args.use_plot, print_step=False)
    elif args.mode.lower() ==  'parallele':
        run_parallel_random_agent(args)
    elif args.mode.lower() == 'episodic':
        run_episodic_random_agent(args)
    elif args.mode.lower() == 'dss':
        run_dss_agent(args)
    else:
        raise NotImplementedError("Running mode not implemented")
