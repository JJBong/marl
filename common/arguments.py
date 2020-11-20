import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--algorithm', type=str, default='qmix')    # vdn, qmix
    parser.add_argument('--optim', type=str, default='rms')  # rms, adam
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--base_net', type=str, default='rnn')  # mlp, rnn
    parser.add_argument('--two_hyper_layers', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--initial_epsilon', type=float, default=1.0)
    parser.add_argument('--grad_norm_clip', type=int, default=10)
    parser.add_argument('--n_agents', type=int, default=2)
    # one_step_payoff_matrix, two_step_payoff_matrix
    parser.add_argument('--env_name', type=str, default='two_step_payoff_matrix')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rnn_hidden_dim', type=int, default=64)
    parser.add_argument('--mlp_hidden_dim', type=int, default=64)
    parser.add_argument('--qmix_hidden_dim', type=int, default=64)
    parser.add_argument('--hyper_hidden_dim', type=int, default=64)
    parser.add_argument('--training_steps', type=int, default=10000)
    parser.add_argument('--playing_steps', type=int, default=500)
    parser.add_argument('--replay_memory_size', type=int, default=1000)
    parser.add_argument('--target_network_update_interval', type=int, default=200)
    parser.add_argument('--print_interval', type=int, default=200)
    parser.add_argument('--cuda', type=bool, default=False)
    args = parser.parse_args()
    return args
