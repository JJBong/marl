from network.base import MLP, RNN
from network.qmix import QMixNet
import torch


class QMix:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.obs_shape = self.args.obs_shape
        input_shape = self.obs_shape

        assert self.args.base_net in ['mlp', 'rnn']
        if self.args.base_net == 'mlp':
            self.eval_base_net = MLP(input_shape, self.args).to(device=self.args.device)
            self.target_base_net = MLP(input_shape, self.args).to(device=self.args.device)
        else:
            self.eval_base_net = RNN(input_shape, self.args).to(device=self.args.device)
            self.target_base_net = RNN(input_shape, self.args).to(device=self.args.device)

        self.target_base_net.load_state_dict(self.eval_base_net.state_dict())

        self.eval_parameters = list(self.eval_base_net.parameters())

    def get_q_value(self, obs, h_in=None):
        if h_in is None:
            q_val = self.eval_base_net(obs)
            return q_val
        else:
            q_val, h_out = self.eval_base_net(obs, h_in)
            return q_val, h_out

    def get_target_q_value(self, obs, h_in=None):
        if h_in is None:
            q_val = self.target_base_net(obs)
            return q_val
        else:
            q_val, h_out = self.target_base_net(obs, h_in)
            return q_val, h_out

    def update_net(self, params=None):
        if params is None:
            self.target_base_net.load_state_dict(self.eval_base_net.state_dict())
        else:
            self.eval_base_net.load_state_dict(params)

    def get_net_params(self):
        return self.eval_base_net.state_dict()


class QMixTrainer:
    def __init__(self, eval_parameters, args):
        self.args = args

        self.eval_qmix_net = QMixNet(self.args)
        self.target_qmix_net = QMixNet(self.args)

        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = eval_parameters + list(self.eval_qmix_net.parameters())

        self.gamma = torch.tensor(self.args.gamma, dtype=torch.float)
        if self.args.optim == 'rms':
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.args.learning_rate)
        elif self.args.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.args.learning_rate)

    def train_agents(self, q_evals, max_q_prime_evals, state, next_state, reward, done_mask):
        q_total_eval = self.eval_qmix_net(q_evals, state)
        max_q_prime_total_eval = self.target_qmix_net(max_q_prime_evals, next_state)

        target = reward + self.gamma * max_q_prime_total_eval * done_mask

        td_error = target.detach() - q_total_eval
        loss = torch.mean((td_error ** 2))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        return loss

    def get_q_value(self, q_evals, state):
        q_total_eval = self.eval_qmix_net(q_evals, state)
        return q_total_eval

    def update_net(self, params=None):
        if params is None:
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        else:
            self.eval_qmix_net.load_state_dict(params)

    def get_net_params(self):
        return self.eval_qmix_net.state_dict()
