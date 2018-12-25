#!/usr/bin/env python
import sys
sys.path.append("../..")
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger


import opensim as osim
from osim.env import *
from osim.http.client import Client

import argparse

def train(num_timesteps, seed, visualize, save_interval, resume):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = RunEnv(visualize)
    env.reset()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear', save_interval=5000, resume=False
        )
    env.close()

def test(model):
    if model is not None:
        print('test')



def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', help='start train', dest='train', action='store_true', default=True)
    parser.add_argument('--test', help='start test', dest='test', action='store_true', default=False)
    parser.add_argument('--num_timesteps', help='the number of timesteps', type=int, default=1e6)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--visualize', help='visualization', dest='visualize', action='store_true', default=False)
    parser.add_argument('--resume', help='resume training', type=str, default=None)
    parser.add_argument('--save_interval', help='save interval', type=int, default=10000)
    parser.add_argument('--model', help='test model', type=str, default=None)
    args = parser.parse_args()

    if args.train:
        train(args.num_timesteps, args.seed, args.visualize, args.save_interval, args.resume)
    if args.test:
        test(args.model)


if __name__ == '__main__':
    main()
