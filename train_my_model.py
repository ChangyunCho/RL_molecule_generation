from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import logger

import tensorflow as tf
import gym

def train(args,seed,writer=None):
    import my_pposgd_simple_gcn
    import my_gcn_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make('molecule-v0')
    env.init(data_type=args.dataset,logp_ratio=args.logp_ratio,qed_ratio=args.qed_ratio,sa_ratio=args.sa_ratio,reward_step_total=args.reward_step_total,is_normalize=args.normalize_adj,reward_type=args.reward_type,
             reward_target=args.reward_target,has_feature=bool(args.has_feature),is_conditional=bool(args.is_conditional),conditional=args.conditional,max_action=args.max_action,min_action=args.min_action)
    env.seed(workerseed)
        
    def policy_fn(name, ob_space, ac_space):
        return my_gcn_policy.GCNPolicy(name=name, ob_space=ob_space, ac_space=ac_space, atom_type_num=env.atom_type_num,args=args)

    my_pposgd_simple_gcn.learn(args, env, policy_fn, max_timesteps=args.num_steps, timesteps_per_actorbatch=256, clip_param=0.2, entcoeff=0.01, optim_epochs=8, optim_stepsize=args.lr, optim_batchsize=32, gamma=1, lam=0.95, schedule='linear')
    env.close()

