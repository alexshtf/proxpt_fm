from golden_section import min_gss
import math


def neg_entr(z):
    if z > 0:
        return z * math.log(z)
    else:
        return 0


def loss_conjugate(z):
    return neg_entr(z) + neg_entr(1 - z)


class ProxPtFMTrainer:
    def __init__(self, fm, step_size):
        # training parameters
        self.b0 = fm.bias
        self.bs = fm.biases
        self.vs = fm.vs
        self.step_size = step_size

        # temporary state for a single learning step
        self.nnz = None                # number of nonzeros
        self.bias_sum = None           # sum of the biases corresponding to the nonzero indicators
        self.vs_nz = None              # embedding vectors of non-zero indicators, stacked as matrix rows
        self.ones_times_vs_nnz = None   # the matrix above multiplied by a matrix of ones.

    def step(self, w_nz, y_hat):
        self.nnz = w_nz.numel()
        self.bias_sum = self.bs[w_nz].sum().item()
        self.vs_nz = self.vs.weight[w_nz, :]
        self.ones_times_vs_nnz = self.vs_nz.sum(dim=0, keepdim=True)

        def q_neg(z):  # neg. of the maximization objective - since the min_gss code minimizes functions.
            return -(self.q_one(y_hat, z) + self.q_two(y_hat, z) - loss_conjugate(z))

        opt_interval = min_gss(q_neg, 0, 1)
        z_opt = sum(opt_interval) / 2

        self.update_biases(w_nz, y_hat, z_opt)
        self.update_vectors(w_nz, y_hat, z_opt)

    def q_one(self, yhat, z):
        return -0.5 * self.step_size * (1 + self.nnz) * (z ** 2) \
               - yhat * (self.bias_sum + self.b0.item()) * z

    def update_biases(self, w_nz, y_hat, z):
        self.bs[w_nz] = self.bs[w_nz] + self.step_size * z * y_hat
        self.b0.add_(self.step_size * z * y_hat)

    def q_two(self, y_hat, z):
        if z == 0:
            return 0

        # solve the linear system - find the optimal vectors
        vs_opt = self.solve_s_inv_system(y_hat, z)

        # compute q_2
        pairwise = (vs_opt.sum(dim=0).square().sum() - vs_opt.square().sum()) / 2  # the pow-of-sum - sum-of-pow trick
        diff_squared = (vs_opt - self.vs_nz).square().sum()
        return (-z * y_hat * pairwise + diff_squared / (2 * self.step_size)).item()

    def update_vectors(self, w_nz, yhat, z):
        # if z = 0 --> we don't need to update the vectors.
        if z == 0:
            return

        # update the vectors with the optimal ones
        self.vs.weight[w_nz, :].sub_(self.vectors_update_dir(yhat, z))

    def solve_s_inv_system(self, y_hat, z):
        return self.vs_nz - self.vectors_update_dir(y_hat, z)

    def vectors_update_dir(self, y_hat, z):
        beta = self.step_size * y_hat * z
        alpha = beta / (1 + beta)
        return alpha * (self.vs_nz - self.ones_times_vs_nnz / (1 + beta * (1 - self.nnz)))
