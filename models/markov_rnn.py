from tqdm import tqdm

import math
import numpy as np
from numpy.random import dirichlet
import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.rmsprop import RMSprop
from torch.autograd import Variable

from utils.time_distributed import TimeDistributed


class MarkovRNN(nn.Module):
    torch.manual_seed(1)

    optimizer_dispatcher = {"sgd": SGD,
                            "rmsprop": RMSprop,
                            "adam": Adam}

    cell_dispatcher = {"rnn": nn.RNNCell,
                       "gru": nn.GRUCell,
                       "lstm": nn.LSTMCell}

    nonlinearity_dispatcher = {"tanh": torch.tanh}

    def __init__(self, input_dim, output_dim, cell_type="rnn",
                 hidden_dim=16, bias=(False, False), init_std=0.1, trunc_len=10, window_len=1,
                 num_state=3, beta=0.9, alpha=0.5, concentration=10, lr=0.01, weight_decay=0, optimizer="sgd",
                 shuffle_flag=False, device='cpu'):

        super(MarkovRNN , self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.init_std = init_std
        self.trunc_len = trunc_len
        self.window_len = window_len
        self.num_state = num_state
        self.alpha = alpha
        self.beta = beta
        self.concentration = concentration
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.shuffle_flag = shuffle_flag
        self.device = device
        self.cell_type = cell_type

        if self.cell_type == "lstm":
            self.num_state_types = 2
        else:
            self.num_state_types = 1

        self.cell_list = self.__init_cells()
        self.out_layer = nn.Linear(hidden_dim, output_dim, bias=bias[1])

        if num_state == 1:
            alpha_vector = np.array([1])
        else:
            alpha_vector = np.array([alpha] + [(1 - alpha) / (num_state - 1)] * (num_state - 1)) * concentration
        psi_ = dirichlet(alpha_vector, num_state)
        self.psi_ = nn.Parameter(torch.tensor([np.roll(psi_[i], i) for i in range(len(psi_))], dtype=torch.float32))

        self.optimizer = self.optimizer_dispatcher[self.optimizer_name](lr=self.lr, weight_decay=weight_decay, params=self.parameters())

        self.all_belief_list = []
        self.all_likelihood_list = []
        self.all_R_list = []

    def __init_cells(self):
        """
        Initializes RNN cells
        :return nn.ModuleList cell_list: module list of RNN cells
        """
        cell_list = []
        for k in range(self.num_state):
            cell = self.cell_dispatcher[self.cell_type](input_size=self.input_dim, hidden_size=self.hidden_dim, bias=self.bias[0])
            cell_list.append(cell)
        cell_list = nn.ModuleList(cell_list)
        return cell_list

    def __init_states(self, batch_size):
        return [Variable(torch.zeros(batch_size, self.hidden_dim).to(self.device), requires_grad=False) for _ in range(self.num_state_types)]

    def __init_belief(self, batch_size):
        """
        Initializes beliefs from uniform distribution, and normalized such that sums to one
        :param int batch_size: batch size
        :return Variable pi: belief (B x K)
        """
        # pi : B x K
        pi = torch.rand(batch_size, self.num_state)
        pi /= pi.sum(dim=1)
        pi = Variable(pi.to(self.device), requires_grad=True)
        return pi

    def __init_R(self):
        """
        Initializes error covariance matrices for each state
        :return Variable R: error covariance (K x Ny x Ny)
        """
        R = Variable(torch.eye(self.output_dim).repeat(self.num_state, 1, 1).to(self.device), requires_grad=False)
        return R

    def forward(self, x, states, pi):
        """
        Forward pass
        :param x: input (B x Nx)
        :param states: [state (B x Nh)]
        :param pi: belief (B x K)
        :return tuple: out (B x Ny), h (B x Nh), out_k (B x Ny x K), h_k (B x Nh x K)
        """
        states_k = [torch.zeros(x.shape[0], self.hidden_dim, self.num_state).to(self.device) for _ in range(self.num_state_types)]  # B x Nh x K
        for k in range(self.num_state):

            if len(states) == 1:
                sub_state_k = self.cell_list[k](x, states[0])
            else:
                sub_state_k = self.cell_list[k](x, states)

            if not isinstance(sub_state_k, tuple):
                sub_state_k = [sub_state_k]
            else:
                sub_state_k = list(sub_state_k)

            for i, state in enumerate(states_k):
                state[:, :, k] = sub_state_k[i]

        out_k = self.out_layer(states_k[0].permute(0, 2, 1)).permute(0, 2, 1)  # B x Ny x K

        if self.device == "cuda":
            states = [torch.bmm(state, pi.unsqueeze(-1)).squeeze(-1) for state in states_k]  # B x Nh
        else:
            states = [torch.zeros(x.shape[0], self.hidden_dim).to(self.device) for _ in range(self.num_state_types)]  # B x Nh
            for b in range(x.shape[0]):
                for i in range(self.num_state_types):
                    states[i][b] = states_k[i][b] @ pi[b]

        out = self.out_layer(states[0])  # B x Ny

        return out, states, out_k, states_k

    def compute_likelihood(self, error_k, R):
        """
        Compute likelihood of each mode
        :param torch.Tensor error_k: error vectors (B x Ny x K)
        :param Variable R: error covariance (K x Ny x Ny)
        :return: torch.Tensor likelihoods (B x K)
        """
        likelihood = torch.zeros(error_k.shape[0], self.num_state)
        det = [torch.det(R[k].clone()) for k in range(self.num_state)]
        inv = [torch.inverse(R[k].clone()) for k in range(self.num_state)]
        for b in range(error_k.shape[0]):
            for k in range(self.num_state):
                loss_ = error_k[b, :, k].unsqueeze(-1)
                l = ((2 * math.pi) ** (-self.num_state / 2)) * torch.sqrt(det[k]) * \
                    torch.exp(- 0.5 * (loss_.t() @ inv[k] @ loss_))[0]
                likelihood[b, k] = torch.clamp(l, min=1e-6, max=10)

        return likelihood

    def update_belief(self, pi, likelihood):
        """
        Update beliefs
        :param Variable pi: belief (B x K)
        :param torch.Tensor likelihoods (B x K)
        :return: Variable pi: updated belief (B x K)
        """
        pi = (pi @ self.psi) * likelihood
        pi /= pi.sum()  # TODO: softmax?

        return pi

    def update_R(self, error_k, R):
        """
        Update error covariance with exponential smoothing
        :param torch.Tensor error_k: error vectors (B x Ny x K)
        :param Variable R: error covariance (K x Ny x Ny)
        :return: torch.Tensor R_out: updated error covariance (K x Ny x Ny)
        """
        R_out = R.clone()
        for k in range(self.num_state):
            loss = error_k.mean(dim=0)[:, k].reshape(-1, 1)
            R_out[k] = self.beta * R[k] + (1 - self.beta) * (loss @ loss.t())

        return R_out

    def fit(self, data, verbose=True):

        loss_list = []
        pred_list = []
        all_R_list = []
        all_belief_list = []
        all_likelihood_list = []

        batch_size = data[0][0].shape[0]
        start_R = self.__init_R()
        self.R = start_R

        start_states = self.__init_states(batch_size)
        start_belief = self.__init_belief(batch_size)
        self.states = start_states
        self.belief = start_belief

        for i in tqdm(range(self.trunc_len, len(data) + 1)):  # data pass

            if self.shuffle_flag:
                ii = np.random.randint(1, len(data))
            else:
                ii = i

            start_idx = max(ii - self.trunc_len, 0)
            inputs, targets = zip(*data[start_idx: ii])

            likelihood_list = []
            states_list = [start_states]
            belief_list = [start_belief]
            R_list = [start_R]

            self.psi = nn.functional.softmax(self.psi_)

            def closure():
                self.zero_grad()

                loss = 0
                pred_tensor_list = []

                for step, inp in enumerate(inputs):  # bptt pass

                    inp = inp.to(self.device)
                    last_step_flag = step == len(inputs) - 1
                    pred, states, pred_k, states_k = self.forward(inp, states_list[-1], belief_list[-1])
                    states_list.append(states)
                    pred_tensor_list.append(pred)
                    target = targets[step].to(self.device)

                    if not self.shuffle_flag:
                        self.states = states

                    if last_step_flag:  # update at the last step

                        loss = torch.mean((pred - target) ** 2)
                        loss.backward(retain_graph=True)

                        loss_list.append(loss.item())
                        pred_list.append(pred.item())

                    error_k = (target.unsqueeze(dim=-1) - pred_k)

                    with torch.no_grad():
                        likelihood_vec = self.compute_likelihood(error_k,  R_list[-1])

                    belief = self.update_belief(belief_list[-1], likelihood_vec)
                    if not self.shuffle_flag:
                        self.belief = belief

                    with torch.no_grad():
                        R = self.update_R(error_k, R_list[-1])
                    if not self.shuffle_flag:
                        self.R = R

                    belief_list.append(belief)
                    R_list.append(R)
                    likelihood_list.append(likelihood_vec)

                    if not self.shuffle_flag and step == 0:  # hold state for next data pass
                        for j, state in enumerate(start_states):
                            state.data = states[j].data
                        start_belief.data = belief.data

                    if not self.shuffle_flag and last_step_flag:
                        start_R.data = R.data

                return loss

            self.optimizer.step(closure)

            all_belief_list.append(belief_list[-1])
            all_R_list.append(R_list[-1])
            all_likelihood_list.append(likelihood_list[-1])

        self.all_belief_list = all_belief_list
        self.all_R_list = all_R_list
        self.all_likelihood_list = all_likelihood_list

        return pred_list, loss_list

    def predict(self, data, run_states=True):

        loss_list = []
        pred_list = []
        pred_tensor_list = []

        states = self.states
        belief = self.belief
        R = self.R
        inputs, targets = zip(*data)

        for step, inp in enumerate(inputs):  # bptt pass

            inp = inp.to(self.device)
            pred, states, pred_k, states_k = self.forward(inp, states, belief)
            pred_tensor_list.append(pred)
            target = targets[step].to(self.device)

            loss = torch.mean((pred - target) ** 2)
            error_k = (target.unsqueeze(dim=-1) - pred_k)

            likelihood_vec = self.compute_likelihood(error_k, R)

            belief = self.update_belief(belief, likelihood_vec)
            R = self.update_R(error_k, R)

            loss_list.append(loss.item())
            pred_list.append(pred.item())

        if run_states:
            self.states = states
            self.belief = belief
            self.R = R

        return pred_list, loss_list
