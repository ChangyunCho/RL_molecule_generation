import os
from train_my_model import train

def arg_parser():
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def molecule_arg_parser():
    parser = arg_parser()
    parser.add_argument('--env', type=str, default='molecule')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--num_steps', type=int, default=int(5e7))
    parser.add_argument('--name', type=str, default='test_conditional')
    parser.add_argument('--name_load', type=str, default='0new_concatno_mean_layer3_expert1500')
    parser.add_argument('--dataset', type=str, default='zinc')
    parser.add_argument('--dataset_load', type=str, default='zinc')
    parser.add_argument('--reward_type', type=str, default='logppen')
    parser.add_argument('--reward_target', type=float, default=0.5)
    parser.add_argument('--logp_ratio', type=float, default=1)
    parser.add_argument('--qed_ratio', type=float, default=1)
    parser.add_argument('--sa_ratio', type=float, default=1)
    parser.add_argument('--gan_step_ratio', type=float, default=1)
    parser.add_argument('--gan_final_ratio', type=float, default=1)
    parser.add_argument('--reward_step_total', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--has_d_step', type=int, default=1)
    parser.add_argument('--has_d_final', type=int, default=1)
    parser.add_argument('--has_ppo', type=int, default=1)
    parser.add_argument('--rl_start', type=int, default=250)
    parser.add_argument('--rl_end', type=int, default=int(1e6))
    parser.add_argument('--expert_start', type=int, default=0)
    parser.add_argument('--expert_end', type=int, default=int(1e6))
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--load_step', type=int, default=250)
    parser.add_argument('--curriculum', type=int, default=0)
    parser.add_argument('--curriculum_num', type=int, default=6)
    parser.add_argument('--curriculum_step', type=int, default=200)
    parser.add_argument('--supervise_time', type=int, default=4)
    parser.add_argument('--normalize_adj', type=int, default=0)
    parser.add_argument('--layer_num_g', type=int, default=3)
    parser.add_argument('--layer_num_d', type=int, default=3)
    parser.add_argument('--graph_emb', type=int, default=0)
    parser.add_argument('--stop_shift', type=int, default=-3)
    parser.add_argument('--has_residual', type=int, default=0)
    parser.add_argument('--has_concat', type=int, default=0)
    parser.add_argument('--has_feature', type=int, default=0)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--gcn_aggregate', type=str, default='mean')
    parser.add_argument('--gan_type', type=str, default='normal')
    parser.add_argument('--gate_sum_d', type=int, default=0)
    parser.add_argument('--mask_null', type=int, default=0)
    parser.add_argument('--is_conditional', type=int, default=0)
    parser.add_argument('--conditional', type=str, default='low')
    parser.add_argument('--max_action', type=int, default=128)
    parser.add_argument('--min_action', type=int, default=20)
    parser.add_argument('--bn', type=int, default=0)
    parser.add_argument('--name_full',type=str, default='')
    parser.add_argument('--name_full_load',type=str, default='')
    return parser

def main():
    from gym.envs.registration import register
    register(id='molecule-v0', entry_point='gym_molecule.envs:MoleculeEnv',)
    args = molecule_arg_parser().parse_args()
    #print(args)
    args.name_full = args.env + '_' + args.dataset + '_' + args.name
    args.name_full_load = args.env + '_' + args.dataset_load + '_' + args.name_load + '_' + str(args.load_step)

    if not os.path.exists('molecule_gen'):
        os.makedirs('molecule_gen')
    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')

    train(args,seed=args.seed)

if __name__ == '__main__':
    main()





