from common.replay_memory import ReplayMemoryForMLP, ReplayMemoryForRNN
from agent.agents import Agents


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        assert self.args.base_net in ['mlp', 'rnn']
        if self.args.base_net == 'mlp':
            self.replay_memory = ReplayMemoryForMLP(self.args)
        else:
            self.replay_memory = ReplayMemoryForRNN(self.args)

        self.training_steps = self.args.training_steps
        self.playing_steps = self.args.playing_steps

        self.agents = Agents(self.args)

    def run(self):
        step = 0
        while step < self.training_steps:
            state, observations = self.env.reset()
            done = False
            while not done:
                actions = self.agents.choose_action(observations)
                next_state, next_observations, reward, done = self.env.step(actions)
                # print('step: {0}, state: {1}, actions: {2}, reward: {3}'.format(step, state, actions, reward))
                done_mask = 0.0 if done else 1.0
                self.replay_memory.put([state, observations, actions, reward, next_state, next_observations, done_mask])

                if self.replay_memory.size() >= self.args.batch_size:
                    s, o, a, r, s_prime, o_prime, done_mask = self.replay_memory.sample(self.args.batch_size)

                    batch = {}
                    batch['state'] = s
                    batch['observation'] = o
                    batch['action'] = a
                    batch['reward'] = r
                    batch['next_state'] = s_prime
                    batch['next_observation'] = o_prime
                    batch['done_mask'] = done_mask

                    loss = self.agents.train(batch, step)

                    if step % self.args.print_interval == 0:
                        print("step: {0}, loss: {1}".format(step, loss))

                state = next_state
                observations = next_observations
                step += 1

                if done:
                    break
        self.agents.save_model()

    def play(self):
        self.agents.load_model()

        q_value_list, iteration, selected_q_value_list, q_value_list_0, q_value_list_1, q_value_list_2, \
        iteration_0, iteration_1, iteration_2 = None, None, None, None, None, None, None, None, None

        if self.args.env_name == 'one_step_payoff_matrix':
            q_value_list = [[0. for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
            iteration = [[0 for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
        elif self.args.env_name == 'two_step_payoff_matrix':
            q_value_list_0 = [[0. for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
            iteration_0 = [[0 for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
            q_value_list_1 = [[0. for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
            iteration_1 = [[0 for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
            q_value_list_2 = [[0. for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
            iteration_2 = [[0 for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
        else:
            raise Exception("Wrong env name.")

        step = 0
        while step < self.playing_steps:
            state, observations = self.env.reset()
            done = False

            state_num = 0
            while not done:
                actions, q_total_evals = self.agents.choose_action(observations, state)
                next_state, next_observations, reward, done = self.env.step(actions)

                state = next_state
                observations = next_observations

                if self.args.env_name == 'one_step_payoff_matrix':
                    q_value_list[actions[0]][actions[1]] += q_total_evals
                    iteration[actions[0]][actions[1]] += 1
                elif self.args.env_name == 'two_step_payoff_matrix':
                    if state_num == 0:
                        if actions[0] == 0:
                            state_num = 1
                        if actions[0] == 1:
                            state_num = 2
                        q_value_list_0[actions[0]][actions[1]] += q_total_evals
                        iteration_0[actions[0]][actions[1]] += 1
                    else:
                        if state_num == 1:
                            q_value_list_1[actions[0]][actions[1]] += q_total_evals
                            iteration_1[actions[0]][actions[1]] += 1
                        elif state_num == 2:
                            q_value_list_2[actions[0]][actions[1]] += q_total_evals
                            iteration_2[actions[0]][actions[1]] += 1

                step += 1

                if done:
                    break

        if self.args.env_name == 'one_step_payoff_matrix':
            for i in range(self.args.n_actions):
                for j in range(self.args.n_actions):
                    q_value_list[i][j] /= iteration[i][j]
            print(q_value_list)
        elif self.args.env_name == 'two_step_payoff_matrix':
            for i in range(self.args.n_actions):
                for j in range(self.args.n_actions):
                    q_value_list_0[i][j] /= iteration_0[i][j]
                    q_value_list_1[i][j] /= iteration_1[i][j]
                    q_value_list_2[i][j] /= iteration_2[i][j]
            print(q_value_list_0)
            print(q_value_list_1)
            print(q_value_list_2)
