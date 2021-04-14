import torch
from torch import nn


class FM(torch.nn.Module):
    def __init__(self, m, k):
        super(FM, self).__init__()

        self.bias = nn.Parameter(torch.zeros(1))
        self.biases = nn.Parameter(torch.zeros(m))
        self.vs = nn.Embedding(m, k)

        with torch.no_grad():
            torch.nn.init.trunc_normal_(self.vs.weight, std=0.01)
            torch.nn.init.trunc_normal_(self.biases, std=0.01)

    def forward(self, w_nz):  # since w are indicators, we simply use the non-zero indices
        vs = self.vs(w_nz)
        # in vs:
        #   dim = 0 is the mini-batch dimension. We would like to operate on each elem. of a mini-batch separately.
        #   dim = 1 are the embedding vectors
        #   dim = 2 are their components.

        pow_of_sum = vs.sum(dim=1).square().sum(dim=1)  # sum vectors -> square -> sum components
        sum_of_pow = vs.square().sum(dim=[1, 2])        # square -> sum vectors and components
        pairwise = 0.5 * (pow_of_sum - sum_of_pow)

        biases = self.biases
        linear = biases[w_nz].sum(dim=1)                # sum biases for each element of the mini-batch

        return pairwise + linear + self.bias
