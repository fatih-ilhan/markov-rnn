import torch.nn as nn


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        """
        Used by CNN to accept multiple input tensor with time axis.

        :param module: Main module using this class.
        :param batch_first: Batch dimension before time dimension or vice versa.
        """
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            # Works only for input tensors larger than 2 dimensions.
            return self.module(x)

        # Squash samples and time steps into a single axis and then feed to the module.
        x_reshape = x.contiguous().view(-1, *x.shape[2:])  # (s

        # samples * timesteps, input_size)
        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, *y.shape[1:])  # (samples, time steps, output_size)
        else:
            y = y.view(-1, x.size(1), *y.shape[1:])  # (time steps, samples, output_size)

        return y
