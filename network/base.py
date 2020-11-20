import torch.nn as nn
import torch.nn.functional as f


class MLP(nn.Module):
    def __init__(self, input_shape, args):
        super(MLP, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.n_actions)

    def forward(self, obs):
        x = f.relu(self.fc1(obs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class RNN(nn.Module):
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        x = x.reshape(-1, self.args.rnn_hidden_dim)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
