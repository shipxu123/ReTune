import math

import torch
import gpytorch
import numpy as np
from torch.nn.modules.module import Module


class MixedDiffusionKernel(gpytorch.kernels.Kernel):
    def __init__(self,
                 log_order_variances,
                 grouped_log_beta,
                 fourier_freq_list,
                 fourier_basis_list,
                 lengthscales,
                 num_discrete,
                 num_continuous):
        self.log_amp = gpytorch.kernels.Kernel.register_parameter(
            name='log_amp',
            parameter=torch.nn.Parameter(torch.FloatTensor(1))
        )
        self.log_order_variances = gpytorch.kernels.Kernel.register_parameter(
            name='log_order_variances',
            parameter=torch.nn.Parameter(log_order_variances)
        )
        self.grouped_log_beta = grouped_log_beta
        self.fourier_freq_list = fourier_freq_list
        self.fourier_basis_list = fourier_basis_list
        self.lengthscales = gpytorch.kernels.Kernel.register_parameter(
            name='lengthscales',
            parameter=torch.nn.Parameter(lengthscales)
        )
        self.num_discrete = num_discrete
        self.num_continuous = num_continuous
        assert self.log_order_variances.size(0) == self.num_continuous + self.num_discrete, "order variances are not properly initialized"
        assert self.lengthscales.size(0) == self.num_continuous, "lengthscales is not properly initialized"
        assert self.grouped_log_beta.size(0) == self.num_discrete, "beta is not properly initialized"

    def n_params(self):
        return 1 

    def param_to_vec(self):
        return self.log_amp.clone()

    def vec_to_param(self, vec):
        assert vec.numel() == 1 # self.num_discrete + self.num_continuous
        self.log_amp = vec[:1].clone()

    def forward(self, x1, x2=None, diagonal=False):
        """
        :param x1, x2: each row is a vector with vertex numbers starting from 0 for each 
        :return: 
        """
        if diagonal:
            assert x2 is None
        stabilizer = 0
        if x2 is None:
            x2 = x1
            if diagonal:
                stabilizer = 1e-6 * x1.new_ones(x1.size(0), 1, dtype=torch.float32)
            else:
                stabilizer = torch.diag(1e-6 * x1.new_ones(x1.size(0), dtype=torch.float32))

        base_kernels = []
        for i in range(len(self.fourier_freq_list)):
            beta = torch.exp(self.grouped_log_beta[i])
            fourier_freq = self.fourier_freq_list[i]
            fourier_basis = self.fourier_basis_list[i]
            cat_i = fourier_freq.size(0)
            discrete_kernel = ((1-torch.exp(-beta*cat_i))/(1+(cat_i-1)*torch.exp(-beta*cat_i)))**((x1[:, i].unsqueeze(1)[:, np.newaxis] != x2[:, i].unsqueeze(1)).sum(axis=-1))
            if diagonal:
                base_kernels.append(torch.diagonal(discrete_kernel).unsqueeze(1))
            else:
                base_kernels.append(discrete_kernel)

        lengthscales = torch.exp(self.lengthscales)**2
        temp_x_1 = x1[:, self.num_discrete:] / lengthscales
        temp_x_2 = x2[:, self.num_discrete:] / lengthscales

        for i in range(self.num_continuous):
            normalized_dists = torch.cdist(temp_x_1[:, i].unsqueeze(1), temp_x_2[:, i].unsqueeze(1))
            gaussian_kernel = torch.exp(-0.5 * (normalized_dists) ** 2)
            if not diagonal:
                base_kernels.append(gaussian_kernel)
            else:
                base_kernels.append(torch.diagonal(gaussian_kernel).unsqueeze(1))
        base_kernels = torch.stack(base_kernels)
        if diagonal:
            base_kernels = base_kernels.squeeze(-1)

        num_dimensions = self.num_discrete + self.num_continuous
        if (not diagonal):
            e_n = torch.empty([num_dimensions + 1, \
                base_kernels.size(1), base_kernels.size(2)])
            e_n[0, :, :] = 1.0
            interaction_orders = torch.arange(1, num_dimensions+1).reshape([-1, 1, 1, 1]).float()
            kernel_dim = -3
            shape = [1 for _ in range(3)]
        else:
            e_n = torch.empty([num_dimensions + 1, \
                base_kernels.size(1)])
            e_n[0, :] = 1.0
            interaction_orders = torch.arange(1, num_dimensions+1).reshape([-1, 1, 1]).float()
            kernel_dim = -2
            shape = [1 for _ in range(2)]

        s_k = base_kernels.unsqueeze(kernel_dim - 1).pow(interaction_orders).sum(dim=kernel_dim)
        m1 = torch.tensor([-1.0])
        shape[kernel_dim] = -1


        for deg in range(1, num_dimensions + 1):  # deg goes from 1 to R (it's 1-indexed!)
            ks = torch.arange(1, deg + 1, dtype=torch.float).reshape(*shape)  # use for pow
            kslong = torch.arange(1, deg + 1, dtype=torch.long)  # use for indexing
            # note that s_k is 0-indexed, so we must subtract 1 from kslong
            sum_ = (
            m1.pow(ks - 1) * e_n.index_select(kernel_dim, deg - kslong) * s_k.index_select(kernel_dim, kslong - 1)
            ).sum(dim=kernel_dim) / deg
            if kernel_dim == -3:
                e_n[deg, :, :] = sum_
            else:
                e_n[deg, :] = sum_

        order_variances = torch.exp(self.log_order_variances)
        if kernel_dim == -3:
            kernel_mat = torch.exp(self.log_amp) * ((order_variances.unsqueeze(-1).unsqueeze(-1) * e_n.narrow(kernel_dim, 1, num_dimensions)).sum(dim=kernel_dim)) + stabilizer
            return torch.exp(self.log_amp) * ((order_variances.unsqueeze(-1).unsqueeze(-1) * e_n.narrow(kernel_dim, 1, num_dimensions)).sum(
                dim=kernel_dim) + stabilizer)
        else:
            return torch.exp(self.log_amp) * ((order_variances.unsqueeze(-1) * e_n.narrow(kernel_dim, 1, num_dimensions)).sum(dim=kernel_dim) + stabilizer)

if __name__ == '__main__':
    dimension = 8
    num_discrete = 6
    num_continuous = 2
    print(f"num_discrete: {num_discrete}, num_continuous: {num_continuous}")

    n_vertices = [4, 4, 3, 5, 4, 5]
    adjacency_mat = []
    fourier_freq = []
    fourier_basis = []

    for i in range(len(n_vertices)):
        n_v = n_vertices[i]
        adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
        adjacency_mat.append(adjmat)
        laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
        eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
        fourier_freq.append(eigval)
        fourier_basis.append(eigvec)

    n_vertices = n_vertices
    adj_mat_list = adjacency_mat
    grouped_log_beta = torch.ones(len(fourier_freq))
    log_order_variances = torch.zeros((num_discrete + num_continuous))
    fourier_freq_list = fourier_freq
    fourier_basis_list = fourier_basis
    lengthscales = torch.zeros(num_continuous)

    kernel = MixedDiffusionKernel(log_order_variances=log_order_variances, grouped_log_beta=grouped_log_beta,
                              fourier_freq_list=fourier_freq_list,
                              fourier_basis_list=fourier_basis_list, lengthscales=lengthscales,
                              num_discrete=num_discrete, num_continuous=num_continuous)

    x1 = torch.ones([1, dimension])
    x2 = torch.zeros([1, dimension])
    print(kernel.forward(x1, x2))