import tensorflow as tf
import numpy as np
import time
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import os
import copy

def learn(args,env, policy_fn, *,
          timesteps_per_actorbatch, # timesteps per actor per update
          clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
          gamma, lam, # advantage estimation
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None, # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          schedule='constant'): # annealing for stepsize parameters (epsilon and adam)
    
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.compat.v1.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = {}
    ob['adj'] = U.get_placeholder_cached(name="adj")
    ob['node'] = U.get_placeholder_cached(name="node")

    ob_gen = {}
    ob_gen['adj'] = U.get_placeholder(shape=[None, ob_space['adj'].shape[0], None, None], dtype=tf.float32,name='adj_gen')
    ob_gen['node'] = U.get_placeholder(shape=[None, 1, None, ob_space['node'].shape[2]], dtype=tf.float32,name='node_gen')

    ob_real = {}
    ob_real['adj'] = U.get_placeholder(shape=[None,ob_space['adj'].shape[0],None,None],dtype=tf.float32,name='adj_real')
    ob_real['node'] = U.get_placeholder(shape=[None,1,None,ob_space['node'].shape[2]],dtype=tf.float32,name='node_real')

    ac = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None,4],name='ac_real')

    ## PPO loss
    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    pi_logp = pi.pd.logp(ac)
    oldpi_logp = oldpi.pd.logp(ac)
    ratio_log = pi.pd.logp(ac) - oldpi.pd.logp(ac)

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    ## Expert loss
    loss_expert = -tf.reduce_mean(pi_logp)
    
    var_list_pi = pi.get_trainable_variables()
    var_list_pi_stop = [var for var in var_list_pi if ('emb' in var.name) or ('gcn' in var.name) or ('stop' in var.name)]

    ## debug
    debug={}

    ## loss update function
    lossandgrad_ppo = U.function([ob['adj'], ob['node'], ac, pi.ac_real, oldpi.ac_real, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list_pi)])
    lossandgrad_expert = U.function([ob['adj'], ob['node'], ac, pi.ac_real], [loss_expert, U.flatgrad(loss_expert, var_list_pi)])
    lossandgrad_expert_stop = U.function([ob['adj'], ob['node'], ac, pi.ac_real], [loss_expert, U.flatgrad(loss_expert, var_list_pi_stop)])

    adam_pi = MpiAdam(var_list_pi, epsilon=adam_epsilon)
    adam_pi_stop = MpiAdam(var_list_pi_stop, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.compat.v1.assign(oldv, newv) for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    
    compute_losses = U.function([ob['adj'], ob['node'], ac, pi.ac_real, oldpi.ac_real, atarg, ret, lrmult], losses)



    # Prepare for rollouts
    # ----------------------------------------
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    lenbuffer_valid = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_env = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_final = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_final_stat = deque(maxlen=100) # rolling buffer for episode rewardsn

    seg_gen = traj_segment_generator(args, pi, env, timesteps_per_actorbatch, True)

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"
    if args.load==1:
        try:
            fname = './ckpt/' + args.name_full_load
            sess = tf.compat.v1.get_default_session()
            saver = tf.compat.v1.train.Saver(var_list_pi)
            saver.restore(sess, fname)
            iters_so_far = int(fname.split('_')[-1])+1
            print('model restored!', fname, 'iters_so_far:', iters_so_far)
        except:
            print(fname,'ckpt not found, start with iters 0')

    U.initialize()
    adam_pi.sync()
    adam_pi_stop.sync()

    counter = 0
    level = 0
    
    ## start training
    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError


        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)
        ob_adj, ob_node, ac, atarg, tdlamret = seg["ob_adj"], seg["ob_node"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
        d = Dataset(dict(ob_adj=ob_adj, ob_node=ob_node, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob_adj.shape[0]


        # inner training loop, train policy
        for i_optim in range(optim_epochs):

            loss_expert=0
            loss_expert_stop=0
            g_expert=0
            g_expert_stop=0
            pretrain_shift = 5
            g_ppo = 0
            
            ## Expert
            if iters_so_far>=args.expert_start and iters_so_far<=args.expert_end+pretrain_shift:
                ## Expert train
                # # # learn how to stop
                ob_expert, ac_expert = env.get_expert(optim_batchsize)
                loss_expert, g_expert = lossandgrad_expert(ob_expert['adj'], ob_expert['node'], ac_expert, ac_expert)
                loss_expert = np.mean(loss_expert)


            ## PPO
            if iters_so_far>=args.rl_start and iters_so_far<=args.rl_end:
                assign_old_eq_new() # set old parameter values to new parameter values
                batch = d.next_batch(optim_batchsize)

                if iters_so_far >= args.rl_start+pretrain_shift:
                    *newlosses, g_ppo = lossandgrad_ppo(batch["ob_adj"], batch["ob_node"], batch["ac"], batch["ac"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    losses_ppo=newlosses

            # update generator
            adam_pi.update(0.2*g_ppo+0.05*g_expert, optim_stepsize * cur_lrmult)

        ## PPO val
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob_adj"],batch["ob_node"], batch["ac"], batch["ac"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)

        lrlocal = (seg["ep_lens"],seg["ep_lens_valid"], seg["ep_rets"],seg["ep_rets_env"],seg["ep_final_rew"],seg["ep_final_rew_stat"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, lens_valid, rews, rews_env, rews_final,rews_final_stat = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        lenbuffer_valid.extend(lens_valid)
        rewbuffer.extend(rews)
        rewbuffer_env.extend(rews_env)
        rewbuffer_final.extend(rews_final)
        rewbuffer_final_stat.extend(rews_final_stat)

        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)

        if MPI.COMM_WORLD.Get_rank() == 0:
            with open('molecule_gen/' + args.name_full + '.csv', 'a') as f:
                f.write('***** Iteration {} *****\n'.format(iters_so_far))
            # save
            if iters_so_far % args.save_every == 0:
                fname = './ckpt/' + args.name_full + '_' + str(iters_so_far)
                saver = tf.compat.v1.train.Saver(var_list_pi)
                saver.save(tf.compat.v1.get_default_session(), fname)
                print('model saved!',fname)

        iters_so_far += 1
        counter += 1
        if counter%args.curriculum_step and counter//args.curriculum_step<args.curriculum_num:
            level += 1

def traj_segment_generator(args, pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()
    ob_adj = ob['adj']
    ob_node = ob['node']

    cur_ep_ret = 0 # return in current episode
    cur_ep_ret_env = 0
    cur_ep_len = 0 # len of current episode
    cur_ep_len_valid = 0
    ep_rets = [] # returns of completed episodes in this segment
    ep_rets_env = []
    ep_lens = [] # lengths of ...
    ep_lens_valid = [] # lengths of ...
    ep_rew_final = []
    ep_rew_final_stat = []



    # Initialize history arrays
    ob_adjs = np.array([ob_adj for _ in range(horizon)])
    ob_nodes = np.array([ob_node for _ in range(horizon)])
    ob_adjs_final = []
    ob_nodes_final = []
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred, debug = pi.act(stochastic, ob)

        if t > 0 and t % horizon == 0:
            yield {"ob_adj" : ob_adjs, "ob_node" : ob_nodes,"ob_adj_final" : np.array(ob_adjs_final), "ob_node_final" : np.array(ob_nodes_final), "rew" : rews, "vpred" : vpreds, "new" : news, "ac" : acs, "prevac" : prevacs,
                   "nextvpred": vpred * (1 - new), "ep_rets" : ep_rets, "ep_lens" : ep_lens, "ep_lens_valid" : ep_lens_valid, "ep_final_rew":ep_rew_final, "ep_final_rew_stat":ep_rew_final_stat,"ep_rets_env" : ep_rets_env}
            ep_rets = []
            ep_lens = []
            ep_lens_valid = []
            ep_rew_final = []
            ep_rew_final_stat = []
            ep_rets_env = []
            ob_adjs_final = []
            ob_nodes_final = []

        i = t % horizon
        ob_adjs[i] = ob['adj']
        ob_nodes[i] = ob['node']
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew_env, new, info = env.step(ac)
        rew_d_step = 0 # default
        if rew_env>0: # if action valid
            cur_ep_len_valid += 1

        rews[i] = rew_env

        cur_ep_ret += rews[i]
        cur_ep_ret_env += rew_env
        cur_ep_len += 1

        if new:
            if args.env=='molecule':
                with open('molecule_gen/'+args.name_full+'.csv', 'a') as f:
                    str = ''.join(['{},']*(len(info)+1))[:-1]+'\n'
                    f.write(str.format(info['smile'], info['reward_valid'], info['reward_qed'], info['reward_sa'], info['final_stat'], rew_env, cur_ep_ret, info['flag_steric_strain_filter'], info['flag_zinc_molecule_filter'], info['stop']))
            ob_adjs_final.append(ob['adj'])
            ob_nodes_final.append(ob['node'])
            ep_rets.append(cur_ep_ret)
            ep_rets_env.append(cur_ep_ret_env)
            ep_lens.append(cur_ep_len)
            ep_lens_valid.append(cur_ep_len_valid)
            ep_rew_final.append(rew_env)
            ep_rew_final_stat.append(info['final_stat'])
            cur_ep_ret = 0
            cur_ep_len = 0
            cur_ep_len_valid = 0
            cur_ep_ret_d_step = 0
            cur_ep_ret_env = 0
            ob = env.reset()
        t += 1

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]
