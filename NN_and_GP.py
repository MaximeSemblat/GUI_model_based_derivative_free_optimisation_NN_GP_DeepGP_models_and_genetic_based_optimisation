#!/usr/bin/env python
# coding: utf-8



import numpy as np




from skopt.space import Space
from skopt.sampler import Lhs, Sobol
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math


class DataHandler:
    def __init__(self):
        # input sample space
        self.space = None #array like of shape (n_inputs,2)
        # input, output, and training variables
        self.x = None # array like of shape (n_samples, n_inputs)
        self.y = None # array like of shape (n_samples, n_outputs)
        self.t = None # array like of shape (n_samples,1)
        # training and testing variables
        self.x_train = None #array like of shape (n_train_samples, n_inputs)
        self.x_test = None # array like of shape (n_test_samples, n_inputs)
        self.y_train = None # array like of shape (n_train_samples, n_outputs)
        self.y_test = None #array like of shape (n_test_samples, n_outputs)
        self.t_train = None #array like of shape (n_train_samples, 1)
        self.t_test = None # array like of shape (n_test_samples, 1)
        # scaled variables
        self.space_ = None #array like of shape (n_inputs,2)
        self.x_ = None #array like of shape (n_samples, n_inputs)
        self.y_ = None #array like of shape (n_samples, n_outputs)
        self.x_train_ = None #array like of shape (n_train_samples, n_inputs)
        self.x_test_ = None #array like of shape (n_test_samples, n_inputs)
        self.y_train_ = None #array like of shape (n_train_samples, n_outputs)
        self.y_test_ = None #array like of shape (n_test_samples, n_outputs)
        # scaling paramters / statistical moments
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.x_train_mean = None
        self.x_train_std = None
        self.y_train_mean = None
        self.y_train_std = None

    def init(self, n_samples, space, n_outputs=1, method='lhs'):
        '''
        n_samples         -       number of inputs samples
        space             -       input space
        n_outputs         -       number of ouput dimensions to initialise
        method            -       sampling method: random, lhs, sobol, grid
        '''

        # save space and initialise outputs and targets
        self.space = space
        self.y = np.zeros((n_samples, n_outputs))
        self.t = np.ones((n_samples, 1))

        if method == 'random':
            mat = np.random.rand(n_samples, len(self.space))
            samples = np.zeros_like(mat)
            for i in range(n_samples):
                for j in range(len(self.space)):
                    samples[i][j] = mat[i][j] * (self.space[j][1] - self.space[j][0]) + self.space[j][0]
            self.x = samples

        elif method == 'lhs':
            lhs = Lhs(criterion='maximin', iterations=1000)
            input_space = Space(self.space)
            lhs_samples = lhs.generate(input_space.dimensions, n_samples)
            self.x = np.array(lhs_samples)

        elif method == 'sobol':
            sobol = Sobol()
            input_space = Space(self.space)
            sobol_samples = sobol.generate(input_space.dimensions, n_samples)
            self.x = np.array(sobol_samples)

        elif method == 'grid':
            m = len(space)
            n = math.ceil(n_samples ** (1/m))
            x1, x2 = np.linspace(*self.space[0], n), np.linspace(*self.space[1], n)
            x1_grid, x2_grid = np.meshgrid(x1, x2)
            grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
            samples = np.array(grid)
            np.random.shuffle(samples)
            self.x = samples[:n_samples]


    def split(self, test_size=0.3):
        # train-test split on x, y, t
        self.x_train, self.x_test, self.y_train, self.y_test, self.t_train, self.t_test = train_test_split(
            self.x, self.y, self.t, test_size=test_size)


    def scale(self):
        # normalise x
        scaler = StandardScaler()
        self.x_ = scaler.fit_transform(self.x)
        self.x_mean, self.x_std = scaler.mean_, scaler.scale_

        # normalise y only on converged data
        y_con = self.y[self.t.ravel() == 1, :]
        scaler.fit(y_con)
        self.y_mean, self.y_std = scaler.mean_, scaler.scale_
        self.y_ = (self.y - self.y_mean) / self.y_std

        if self.x_train is not None:
            # normalise x_train
            self.x_train_ = scaler.fit_transform(self.x_train)
            self.x_train_mean, self.x_train_std = scaler.mean_, scaler.scale_
            # normalise x_test using training moments
            self.x_test_ = (self.x_test - self.x_train_mean) / self.x_train_std
            # normalise y_train only on converged data
            y_train_con = self.y_train[self.t_train.ravel() == 1, :]
            scaler.fit(y_train_con)
            self.y_train_mean, self.y_train_std = scaler.mean_, scaler.scale_
            self.y_train_ = (self.y_train - self.y_train_mean) / self.y_train_std
            # normalise y_test using training moments
            self.y_test_ = (self.y_test - self.y_train_mean) / self.y_train_std
            # normalise space using training moments
            self.space_ = []
            for i, val in enumerate(self.space):
                lb = (val[0] - self.x_train_mean[i]) / self.x_train_std[i]
                ub = (val[1] - self.x_train_mean[i]) / self.x_train_std[i]
                self.space_.append( (lb, ub) )
        else:
            # normalise space using x moments
            self.space_ = []
            for i, val in enumerate(self.space):
                lb = (val[0] - self.x_mean[i]) / self.x_std[i]
                ub = (val[1] - self.x_mean[i]) / self.x_std[i]
                self.space_.append( (lb, ub) )


    def scale_space(self, space):
        # normalise space using x moments
        new_space = []
        for i, val in enumerate(space):
            if self.x_train is not None:
                lb = (val[0] - self.x_train_mean[i]) / self.x_train_std[i]
                ub = (val[1] - self.x_train_mean[i]) / self.x_train_std[i]
            else:
                lb = (val[0] - self.x_mean[i]) / self.x_std[i]
                ub = (val[1] - self.x_mean[i]) / self.x_std[i]
            new_space.append( [lb, ub] )
        return new_space


    def inv_scale_x(self, x):
        output = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if self.x_train is not None:
                    output[i, j] = x[i, j] * self.x_train_std[j] + self.x_train_mean[j]
                else:
                    output[i, j] = x[i, j] * self.x_std[j] + self.x_mean[j]
        return output


    def scale_x(self, x):
        output = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if self.x_train is not None:
                    output[i, j] = (x[i, j] - self.x_train_mean[j]) / self.x_train_std[j]
                else:
                    output[i, j] = (x[i, j] - self.x_mean[j]) / self.x_std[j]
        return output


    def inv_scale_y(self, y):
        output = np.zeros_like(y)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if self.y_train is not None:
                    output[i, j] = y[i, j] * self.y_train_std[j] + self.y_train_mean[j]
                else:
                    output[i, j] = y[i, j] * self.y_std[j] + self.y_mean[j]
        return output


    def scale_y(self, y):
        output = np.zeros_like(y)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if self.x_train is not None:
                    output[i, j] = (y[i, j] - self.y_train_mean[j]) / self.y_train_std[j]
                else:
                    output[i, j] = (y[i, j] - self.y_mean[j]) / self.y_std[j]
        return output



import torch
import torch.nn as nn


class SERF(torch.autograd.Function):
    """The SERF activation function."""

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        result = data * torch.erf(torch.log(1 + data.exp()))
        ctx.save_for_backward(data, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Performs a backpropagation."""

        data, result = ctx.saved_tensors
        p = 2.0 / torch.pi**0.5 * (-(1 + data.exp()).log().square()).exp()
        swish = nn.SiLU()
        grad = p * swish(data) + result / data
        return grad_output * grad



#neural network implementation

import numpy as np
import torch
from torch import nn
import time

class SERF(torch.autograd.Function):
    """The SERF activation function."""

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        result = data * torch.erf(torch.log(1 + data.exp()))
        ctx.save_for_backward(data, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Performs a backpropagation."""

        data, result = ctx.saved_tensors
        p = 2.0 / torch.pi**0.5 * (-(1 + data.exp()).log().square()).exp()
        swish = nn.SiLU()
        grad = p * swish(data) + result / data
        return grad_output * grad

class CustomSERF(nn.Module):
    def forward(self, input):
        return SERF.apply(input)

class NN(nn.Sequential):
    def __init__(self, layers, activation='tanh', is_classifier=False, leaky_relu_slope=0.1):
        if is_classifier:
            self.name = 'NNClf'
        else:
            self.name = 'NN'
        self.layers = layers
        self.activation = activation
        self.weights = []
        self.biases = []


        super().__init__(*self._build_layers(layers))
        self.leaky_relu_slope = nn.Parameter(torch.tensor(leaky_relu_slope, requires_grad=True))

    def fit(
        self,
        x,
        y,
        batch_size=10,
        epochs=1000,
        learning_rate=1e-2,
        weight_decay=0.0,
        loss_func=nn.MSELoss(),
        iprint=False
    ):
        if self.name == 'NNClf':
            loss_func = nn.BCEWithLogitsLoss()
        x_train, y_train = torch.Tensor(x), torch.Tensor(y)
        optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train()
        start_time = time.time()
        for epoch in range(epochs):
            permutation = torch.randperm(len(x_train))
            for i in range(0, len(x_train), batch_size):
                idx = permutation[i:i+batch_size]
                x_batch, y_batch = x_train[idx], y_train[idx]
                predictions = self.forward(x_batch)
                loss = loss_func(predictions, y_batch)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()


        end_time = time.time()
        self._get_params()
        if iprint:
            print('{} model fitted! Time elapsed {:.5f} s'.format(self.name, end_time - start_time))
    def _optimize_negative_slope(self, x, y, learning_rate):
        x, y = torch.Tensor(x), torch.Tensor(y)
        optimizer = torch.optim.SGD([self.leaky_relu_slope], lr=learning_rate)
        criterion = nn.MSELoss()

        for _ in range(10000):  # You can adjust the number of iterations
            self.train()
            predictions = torch.sigmoid(self.forward(x)).squeeze()
            loss = criterion(predictions, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, x, return_proba=False, return_class=False, threshold=0.5):
        x = torch.Tensor(x)
        self.eval()
        y = self.forward(x).detach()
        c = torch.max(y, torch.tensor([0.]))
        proba = torch.sigmoid(y).detach()
        c[proba > threshold] = 1
        if return_class and return_proba:
            return y.numpy(), proba.numpy(), c.numpy()
        elif return_class:
            return y.numpy(), c.numpy()
        elif return_proba:
            return y.numpy(), proba.numpy()
        else:
            return y.numpy()

    def formulation(self, x):
        output = np.zeros((x.shape[0], 1))

        for ind, x_val in enumerate(x):

            w = self.weights
            b = self.biases
            af = self._af_selector()
            n = {i: set(range(w[i].shape[1])) for i in range(len(w))}
            a = {key: lambda x: af(x) for key in range(len(w) - 1)}
            a[len(a)] = lambda x: x

            def f(i):
                if i == -1:
                    return x_val
                return a[i](sum(torch.from_numpy(w[i])[:, j] * f(i-1)[j] for j in n[i]) + torch.from_numpy(b[i]))

            output_val = f(len(self.weights) - 1)
            output[ind] = output_val.numpy()

        return output

    def _get_params(self):
        for layer in self:
            if isinstance(layer, nn.Linear):
                self.weights.append(layer.weight.data.numpy())
                self.biases.append(layer.bias.data.numpy())

    def _af_selector(self):
        if self.activation == 'tanh':
            def f(x):
                return 1 - 2 / (np.exp( 2 * x ) + 1)

        elif self.activation == 'sigmoid':
            def f(x):
                return 1 / (1 + np.exp( -x ))
        elif self.activation == 'softplus':
            def f(x):
                return np.log(1 + np.exp(x))

        elif self.activation == 'relu':
            def f(x):
                return np.maximum(0, x)

        elif self.activation == 'linear':
            def f(x):
                return x

        elif self.activation == 'hardsigmoid':
            def f(x):
                y = np.zeros_like(x)
                for i in range(len(y)):
                    if x[i] >= 3:
                        y[i] = 1
                    elif x[i] <= -3:
                        y[i] = 0
                    else:
                        y[i] = x[i] / 6 + 0.5
                return y
        elif self.activation == 'leaky relu':
            def f(x):
              alpha = 0.10000000149011612
              return np.maximum(-alpha*x,x)


        return f

    def _activation_selector(self):
        if self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'softplus':
            return nn.Softplus()
        elif self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'hardsigmoid':
            return nn.Hardsigmoid()
        elif self.activation == 'linear':
            return nn.Identity()
        elif self.activation == 'leaky relu':
            return nn.LeakyReLU(negative_slope=0.10000000149011612)
        elif self.activation == 'swish':
            return nn.SiLU()
        elif self.activation == 'serf':
            return CustomSERF()

    def _build_layers(self, layers):
        torch_layers = []
        for i in range(len(layers) - 2):
            torch_layers.append( nn.Linear(layers[i], layers[i + 1]) )
            torch_layers.append( self._activation_selector() )
        torch_layers.append( nn.Linear(layers[-2], layers[-1]) )
        return torch_layers



# gaussian process initialisation
import numpy as np
from numpy.linalg import inv, slogdet
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, RationalQuadratic, ExpSineSquared, Product, Matern, Sum
import time


class GPR(GaussianProcessRegressor):
    def __init__(self, kernel='rbf', noise=1e-10,optimizer='fmin_l_bfgs_b',n_restarts_optimizer=10, porder=2):
        self.name = 'GPR'
        self.kernel_name = kernel
        self.noise = noise
        self.x_train = None
        self.length_scale = None
        self.constant_value = None
        self.sigma_0 = None
        self.inv_K = None
        self.porder = porder
        super().__init__(kernel=self._kernel(kernel), alpha=noise)

    def _kernel(self, kernel):
        if kernel == 'rbf':
            kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0, 1e5))
        elif kernel == 'linear':
            kernel = 1.0 * DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 1e5))
        elif kernel == 'polynomial':
            kernel = 1.0 * DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 1e5)) ** self.porder
        elif kernel == 'RationalQuadratic':
            kernel = 1.0 * RationalQuadratic(length_scale = 1.0, length_scale_bounds=(0, 1e5))
        elif kernel == 'ExpSineSquared':
            kernel = 1.0 * ExpSineSquared(length_scale=1.0, length_scale_bounds=(0, 1e5))
        elif kernel == 'Matern':
            kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(0, 1e5), nu = 0.1)
        elif kernel == 'Sum_RBF':
            kernel = 1.0* Sum(RBF(length_scale=1.0, length_scale_bounds=(0, 1e5)),RBF(length_scale=1.0, length_scale_bounds=(0, 1e5)))
        elif kernel == 'Sum_RQ':
            kernel = 1.0* Sum(RationalQuadratic(length_scale = 1.0, length_scale_bounds=(0, 1e5)),RationalQuadratic(length_scale = 1.0, length_scale_bounds=(0, 1e5)))
        return kernel

    def fit(self, x, y, iprint=False):
        self.x_train = x
        with np.errstate(divide='ignore'):
            start_time = time.time()
            super().fit(x, y)
            end_time = time.time()
        self._save_params()
        if iprint:
            print('{} model fitted! Time elapsed {:.5f} s'.format(self.name, end_time - start_time))

    def _save_params(self):
        params = self.kernel_.get_params()
        self.constant_value = params['k1__constant_value']
        if self.kernel_name == 'rbf':
            self.length_scale = params['k2__length_scale']
        if self.kernel_name == 'linear':
            self.sigma_0 = params['k2__sigma_0']
        if self.kernel_name == 'polynomial':
            self.sigma_0 = params['k2__kernel__sigma_0']
        self.alpha = self.alpha_.ravel()
        K = self.kernel_(self.x_train, self.x_train) + np.eye(self.x_train.shape[0]) * self.noise
        self.inv_K = inv(K)

    def predict(self, x, return_std=False, return_cov=False):
        if return_std:
            std = super().predict(x, return_std=True)
            return std
        elif return_cov:
            return super().predict(x, return_cov=True)
        else:
            return super().predict(x, return_std=False)

    def formulation(self, x, return_std=False):
        n = self.x_train.shape[0]   # number of training samples
        m = self.x_train.shape[1]   # number of input dimensions
        # squared exponential kernel evaluated at training and new inputs

        if self.kernel_name == 'rbf':
            k = self.constant_value * np.exp(
                -sum(0.5 / self.length_scale ** 2 * (
                    x[:, j].reshape(1, -1) - self.x_train[:, j].reshape(-1, 1)
                    ) ** 2 for j in range(m))
                )
        if self.kernel_name == 'linear':
            k = self.constant_value * (
                self.sigma_0 ** 2 + sum(x[:, j].reshape(1, -1) * self.x_train[:, j].reshape(-1, 1) for j in range(m))
            )
        if self.kernel_name == 'polynomial':
            k = self.constant_value * (
                self.sigma_0 ** 2 + sum(x[:, j].reshape(1, -1) * self.x_train[:, j].reshape(-1, 1) for j in range(m))
            ) ** self.porder
        # linear predictor of mean function
        pred = sum(k[i] * self.alpha[i] for i in range(n)).reshape(-1, 1)
        if return_std:
            # vector-matrix-vector product of k^T K^-1 k
            vMv = sum(
                k[i] * sum(
                    self.inv_K[i, j] * k[j] for j in range(n)
                    ) for i in range(n)
                )
            # variance and std at new input
            if self.kernel_name == 'rbf':
                k_ss = np.array(self.constant_value).reshape(1, 1)
            elif self.kernel_name == 'linear':
                k_ss = self.constant_value * (
                    self.sigma_0 ** 2 + sum(x[:, j].reshape(1, -1) * x[:, j].reshape(-1, 1) for j in range(m))
                )
            elif self.kernel_name == 'polynomial':
                k_ss = self.constant_value * (
                    self.sigma_0 ** 2 + sum(x[:, j].reshape(1, -1) * x[:, j].reshape(-1, 1) for j in range(m))
                ) ** self.porder
            var = np.diag(k_ss) - vMv
            std = np.sqrt(var)
            return pred, std
        else:
            return pred


class GPC:
    def __init__(self):
        self.name = 'GPC'
        self.x_train = None
        self.t_train = None
        self.l = None
        self.sigma_f = None
        self.delta = None
        self.inv_P = None

    def _kernel(self, x1, x2):
        sq_dist = sum(
            (x1[:, j].reshape(1, -1) - x2[:, j].reshape(-1, 1)) ** 2 for j in range(x1.shape[1])
            )
        sq_exp = self.sigma_f ** 2 * np.exp( - 0.5 / self.l ** 2 * sq_dist )
        return sq_exp

    def fit(self, x, t, iprint=False):
        self.x_train = x
        self.t_train = t
        self._calculate_params(iprint=iprint)

    def predict(self, x, return_std=False, return_class=False, threshold=0.5):
        a = self._posterior_mode()
        k_s = self._kernel(x, self.x_train)
        mu = k_s.T.dot(self.t_train - self._sigmoid(a))
        var = self.sigma_f ** 2 - k_s.T.dot(self.inv_P).dot(k_s)
        var = np.diag(var).clip(min=0).reshape(-1, 1)
        beta = np.sqrt(1 + 3.1416 / 8 * var)
        prediction = self._sigmoid(mu / beta)
        if return_class:
            c = prediction.copy()
            c[c >= threshold] = 1
            c[c < threshold] = 0
            return prediction, c
        elif return_std:
            return prediction, np.sqrt(var)
        else:
            return prediction

    def formulation(self, x):
        n = self.x_train.shape[0]
        m = self.x_train.shape[1]
        sq_exp = np.exp(
            -sum(0.5 / self.l ** 2 * (
                x[:, j].reshape(1, -1) - self.x_train[:, j].reshape(-1, 1)) ** 2 for j in range(m))
            )
        mu = self.sigma_f ** 2 * sum(self.delta[i] * sq_exp[i] for i in range(n))
        var = self.sigma_f ** 2 * (1 - sum(
            sq_exp[i] * self.sigma_f ** 2 * sum(
                sq_exp[i_] * self.inv_P[i, i_] for i_ in range(n)) for i in range(n)))
        beta = np.sqrt(1 + 3.1416 / 8 * var)
        prediction = 1 / (1 + np.exp(- mu / beta))
        return prediction.reshape(-1, 1)

    def _posterior_mode(self, max_iter=10, tol=1e-9):
        K = self._kernel(self.x_train, self.x_train)
        a = np.zeros_like(self.t_train)
        I = np.eye(self.x_train.shape[0])
        for i in range(max_iter):
            W = self._sigmoid(a) * (1 - self._sigmoid(a))
            W = np.diag(W.ravel())
            inv_Q = inv(I + W @ K)
            a_new = (K @ inv_Q).dot(self.t_train - self._sigmoid(a) + W.dot(a))
            a_diff = np.abs(a_new - a)
            a = a_new
            if not np.any(a_diff > tol):
                break
        return a

    def _calculate_params(self, iprint):
        start_time = time.time()
        params = minimize(
            fun=self._opt_fun,
            x0=[1.0, 1.0],
            bounds=[(1e-6, None), (1e-6, None)],
            method='L-BFGS-B',
            options={'iprint': -1})
        end_time = time.time()
        self.l = params.x[0]
        self.sigma_f = params.x[1]
        a = self._posterior_mode()
        W = self._sigmoid(a) * (1 - self._sigmoid(a))
        I = np.eye(self.x_train.shape[0])
        W = np.diag(W.ravel()) + 1e-5 * I
        K = self._kernel(self.x_train, self.x_train)
        P = inv(W) + K
        self.inv_P = inv(P)
        self.delta = self.t_train - self._sigmoid(a)
        if iprint:
            print('{} model fitted! Time elapsed {:.5f} s'.format(self.name, end_time - start_time))

    def _opt_fun(self, theta):
        I = np.eye(self.x_train.shape[0])
        self.l = theta[0]
        self.sigma_f = theta[1]
        K = self._kernel(self.x_train, self.x_train) + 1e-5 * I
        inv_K = inv(K)
        a = self._posterior_mode()
        W = self._sigmoid(a) * (1 - self._sigmoid(a))
        W = np.diag(W.ravel())
        ll = self.t_train.T.dot(a) - np.sum(np.log(1.0 + np.exp(a))) - 0.5 * (
            a.T.dot(inv_K).dot(a) +
            slogdet(K)[1] +
            slogdet(W+inv_K)[1])
        return -ll.ravel()

    @staticmethod
    def _sigmoid(a):
        return 1 / (1 + np.exp(-a))



