from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.rmsprop import RMSprop
from torch.autograd import Variable


class RNN(nn.Module):
    torch.manual_seed(1)

    optimizer_dispatcher = {"sgd": SGD,
                            "rmsprop": RMSprop,
                            "adam": Adam}

    cell_dispatcher = {"rnn": nn.RNNCell,
                       "gru": nn.GRUCell,
                       "lstm": nn.LSTMCell}

    def __init__(self, input_dim, output_dim, cell_type="rnn",
                 hidden_dim=16, bias=[0, 0], init_std=0.1, trunc_len=10, window_len=1, lr=0.01,
                 weight_decay=1e-4, optimizer="sgd", shuffle_flag=False, device='cpu'):

        super(RNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.init_std = init_std
        self.trunc_len = trunc_len
        self.window_len = window_len
        self.weight_decay = weight_decay
        self.lr = lr
        self.optimizer_name = optimizer
        self.shuffle_flag = shuffle_flag
        self.device = device
        self.cell_type = cell_type

        if self.cell_type == "lstm":
            self.num_state_types = 2
        else:
            self.num_state_types = 1

        self.cell = self.cell_dispatcher[cell_type](input_size=input_dim, hidden_size=hidden_dim, bias=bias[0])
        self.out_layer = nn.Linear(hidden_dim, output_dim, bias=bias[1])

        self.optimizer = self.optimizer_dispatcher[self.optimizer_name](lr=self.lr, weight_decay=self.weight_decay, params=self.parameters())

    def __init_states(self):
        return [Variable(torch.zeros(1, self.hidden_dim).to(self.device), requires_grad=False) for _ in range(self.num_state_types)]

    def forward(self, x, states):

        if len(states) == 1:
            states = self.cell(x, states[0])
        else:
            states = self.cell(x, states)

        if not isinstance(states, tuple):
            states = [states]
        else:
            states = list(states)

        out = self.out_layer(states[0])
        return out, states

    def fit(self, data, verbose=True):

        loss_list = []
        pred_list = []

        start_states = self.__init_states()
        self.states = start_states

        for i in tqdm(range(1, len(data) + 1)):  # data pass

            if self.shuffle_flag:
                ii = np.random.randint(1, len(data))
            else:
                ii = i

            start_idx = max(ii - self.trunc_len, 0)
            inputs, targets = zip(*data[start_idx: ii])

            states_list = [start_states]

            def closure():
                self.zero_grad()

                loss = 0
                pred_tensor_list = []

                for step, inp in enumerate(inputs):  # bptt pass

                    pred, states = self.forward(inp, states_list[-1])
                    states_list.append(states)
                    pred_tensor_list.append(pred)

                    if not self.shuffle_flag:
                        self.states = states

                    if step == len(inputs) - 1:  # update at the last step
                        pred_tensor = torch.cat(pred_tensor_list[-self.window_len:])
                        target_tensor = torch.cat(targets[-self.window_len:])

                        loss = torch.mean((pred_tensor - target_tensor) ** 2)
                        last_loss = torch.mean((pred_tensor[-1] - target_tensor[-1]) ** 2)

                        loss.backward(retain_graph=False)

                        loss_list.append(last_loss.item())
                        pred_list.append(pred.item())

                    if not self.shuffle_flag and step == 0:  # hold state for next data pass
                        for j, state in enumerate(start_states):
                            state.data = states[j].data

                return loss

            self.optimizer.step(closure)

        return pred_list, loss_list

    def predict(self, data, run_states=True):

        loss_list = []
        pred_list = []
        pred_tensor_list = []
        target_tensor_list = []

        inputs, targets = zip(*data)
        states = self.states

        for step, inp in enumerate(inputs):  # bptt pass

            pred, states = self.forward(inp, states)
            pred_tensor_list.append(pred)
            target_tensor_list.append(targets[step])

            pred_tensor = torch.cat(pred_tensor_list[-self.window_len:])
            target_tensor = torch.cat(target_tensor_list[-self.window_len:])

            loss = torch.mean((pred_tensor[-1] - target_tensor[-1]) ** 2)

            loss_list.append(loss.item())
            pred_list.append(pred.item())

        if run_states:
            self.states = states

        return pred_list, loss_list
