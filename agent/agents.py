from algorithm.vdn import VDN, VDNTrainer
from algorithm.qmix import QMix, QMixTrainer
import random
import torch
import os, sys, glob

idx = os.getcwd().index("marl")
PROJECT_HOME = os.getcwd()[:idx]
sys.path.append(PROJECT_HOME)

model_save_path = os.path.join(PROJECT_HOME, "marl", "saved_models")
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)


class Agents:
    def __init__(self, args):
        self.args = args
        self.epsilon = self.args.initial_epsilon

        self.agents = []
        eval_parameters = []

        assert self.args.algorithm in ['vdn', 'qmix']

        for agent_id in range(self.args.n_agents):
            if self.args.algorithm == 'vdn':
                agent = VDN(agent_id, self.args)
                self.agents.append(agent)
                eval_parameters += agent.eval_parameters
                self.trainer = VDNTrainer(eval_parameters, self.args)
            elif self.args.algorithm == 'qmix':
                agent = QMix(agent_id, self.args)
                self.agents.append(agent)
                eval_parameters += agent.eval_parameters
                self.trainer = QMixTrainer(eval_parameters, self.args)

    def choose_action(self, observation, h_in=None, state=None):
        actions = []
        q_evals = []
        h_outs = []
        for a in range(self.args.n_agents):
            obs = observation[self.agents[a].agent_id]
            obs = torch.from_numpy(obs).float()
            obs = obs.unsqueeze(0)
            if self.args.base_net == 'rnn':
                _h_in = h_in[self.agents[a].agent_id]
                _h_in = _h_in.unsqueeze(0)
                q_eval, h_out = self.agents[a].get_q_value(obs, _h_in)
                h_outs.append(h_out)
            else:
                q_eval = self.agents[a].get_q_value(obs)
            action = self.choose_action_with_epsilon_greedy(q_eval)
            action = torch.tensor([action])
            action = action.unsqueeze(0)
            actions.append(action)

            q_eval = q_eval.gather(1, action)
            q_evals.append(q_eval)

        if self.args.play:
            q_evals = torch.stack(q_evals)
            state = torch.tensor(state, dtype=torch.float)
            if self.args.algorithm == 'vdn':
                q_total_eval = self.trainer.get_q_value(q_evals)
            elif self.args.algorithm == 'qmix':
                q_total_eval = self.trainer.get_q_value(q_evals, state)
            else:
                q_total_eval = None
            if self.args.base_net == 'rnn':
                h_outs = torch.stack(h_outs)
                return actions, h_outs, q_total_eval.item()
            else:
                return actions, q_total_eval.item()
        else:
            if self.args.base_net == 'rnn':
                h_outs = torch.stack(h_outs)
                return actions, h_outs
            else:
                return actions

    def choose_action_with_epsilon_greedy(self, q_val):
        coin = random.random()
        if coin < self.epsilon:
            return random.randint(0, self.args.n_actions - 1)
        else:
            return q_val.argmax().item()

    def train(self, batch, step):
        q_evals = []
        max_q_prime_evals = []
        for a in range(self.args.n_agents):
            obs = batch['observation'][:, a]
            obs_prime = batch['next_observation'][:, a]
            action = batch['action'].squeeze(1).unsqueeze(2)[:, a]

            if self.args.base_net == 'rnn':
                _h_in = batch['hidden_in'][:, a]
                q_eval, _ = self.agents[a].get_q_value(obs, _h_in)
            else:
                q_eval = self.agents[a].get_q_value(obs)
            q_eval = q_eval.gather(1, action)
            q_evals.append(q_eval)

            if self.args.base_net == 'rnn':
                _h_out = batch['hidden_out'][:, a]
                max_q_prime_eval, _ = self.agents[a].get_target_q_value(obs_prime, _h_out)
                max_q_prime_eval = max_q_prime_eval.max(1)[0].unsqueeze(1)
            else:
                max_q_prime_eval = self.agents[a].get_target_q_value(obs_prime).max(1)[0].unsqueeze(1)
            max_q_prime_evals.append(max_q_prime_eval)

        q_evals = torch.stack(q_evals, dim=1)
        max_q_prime_evals = torch.stack(max_q_prime_evals, dim=1)

        state = batch['state']
        next_state = batch['next_state']
        reward = batch['reward']
        done_mask = batch['done_mask']

        if self.args.algorithm == 'vdn':
            loss = self.trainer.train_agents(q_evals, max_q_prime_evals, reward, done_mask)
        elif self.args.algorithm == 'qmix':
            loss = self.trainer.train_agents(q_evals, max_q_prime_evals, state, next_state, reward, done_mask)
        else:
            loss = 0.0

        if step > 0 and step % self.args.target_network_update_interval == 0:
            for a in range(self.args.n_agents):
                self.agents[a].update_net()
            self.trainer.update_net()

        return loss

    def save_model(self):
        model_params = {}
        for a in range(self.args.n_agents):
            model_params['agent_{}'.format(str(a))] = self.agents[a].get_net_params()
        model_params['mixer'] = self.trainer.get_net_params()

        model_save_filename = os.path.join(
            model_save_path, "{0}_{1}.pth".format(
                self.args.algorithm, self.args.base_net
            )
        )
        torch.save(model_params, model_save_filename)

    def load_model(self):
        saved_model = glob.glob(os.path.join(
            model_save_path, "{0}_{1}.pth".format(
                self.args.algorithm, self.args.base_net
            )
        ))
        model_params = torch.load(saved_model[0])

        for a in range(self.args.n_agents):
            self.agents[a].update_net(model_params['agent_{}'.format(str(a))])
        self.trainer.update_net(model_params['mixer'])
