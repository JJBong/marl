from env.one_step_payoff_matrix import OneStepPayOffMatrix
from env.two_step_payoff_matrix import TwoStepPayOffMatrix
from runner.runner import Runner
from common.arguments import get_args

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32


def main():
    args = get_args()

    # One Step Pay-off Matrix or Two Step Pay-off Matrix
    if args.env_name == 'one_step_payoff_matrix':
        args.state_shape = 2
        args.obs_shape = 2
        args.n_actions = 3
        value_list = [10.4, 0., 10., 0., 10., 10., 10., 10., 10.]
        env = OneStepPayOffMatrix(value_list=value_list)
    elif args.env_name == 'two_step_payoff_matrix':
        args.state_shape = 4
        args.obs_shape = 4
        args.n_actions = 2
        value_list = [[7., 7., 7., 7.], [0., 1., 1., 8.]]
        env = TwoStepPayOffMatrix(value_list=value_list)
    else:
        raise Exception("Wrong env name.")
    print()
    print("* Environment Name: {}".format(args.env_name))
    print("* Initial Value List: {}".format(value_list))
    print()

    runner = Runner(env, args)
    if args.play:
        runner.play()
    else:
        runner.run()


if __name__ == '__main__':
    main()
