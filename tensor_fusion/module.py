import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.normal import Normal
from .distribution import LogUniform

class TensorFusion(nn.Module):

    def __init__(self, input_sizes, output_size, dropout=0.0, bias=True, device=None, dtype=None):

        super().__init__()

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        
        # initialize weight tensor
        tensorized_shape = input_sizes + (output_size,)
        self.weight_tensor = nn.Parameter(torch.empty(tensorized_shape, device=device, dtype=dtype))
        nn.init.xavier_normal_(self.weight_tensor)

        # initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros((output_size,), device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, inputs):

        fusion_tensor = inputs[0]
        for x in inputs[1:]:
            fusion_tensor = torch.einsum('n...,na->n...a', fusion_tensor, x)
        
        fusion_tensor = self.dropout(fusion_tensor)

        output = torch.einsum('n...,...o->no', fusion_tensor, self.weight_tensor)

        if self.bias is not None:
            output = output + self.bias

        output = F.relu(output)
        output = self.dropout(output)

        return output

class LowRankFusion(nn.Module):

    def __init__(self, input_sizes, output_size, rank, dropout=0.0, bias=True, device=None, dtype=None):

        super().__init__()

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.rank = rank
        self.dropout = nn.Dropout(dropout)

        # initialize weight tensor factors
        factors = [nn.Parameter(torch.empty((input_size, rank), device=device, dtype=dtype)) \
            for input_size in input_sizes]
        factors = factors + [nn.Parameter(torch.empty((output_size, rank), device=device, dtype=dtype))]
        
        for factor in factors:
            nn.init.xavier_normal_(factor)

        self.weight_tensor_factors = nn.ParameterList(factors)

        # initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros((output_size,), device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, inputs):
        
        # tensorized forward propagation
        output = 1.0
        for x, factor in zip(inputs, self.weight_tensor_factors[:-1]):
            output = output * (x @ factor)

        output = output @ self.weight_tensor_factors[-1].T

        if self.bias is not None:
            output = output + self.bias

        output = F.relu(output)
        output = self.dropout(output)

        return output

class AutoRankFusion(nn.Module):

    def __init__(self, input_sizes, output_size, dropout=0.0, bias=True,
                 max_rank=20, prior_type='log_uniform', eta=None, 
                 device=None, dtype=None):
        '''
        args:
            input_sizes: a tuple of ints, (input_size_1, input_size_2, ..., input_size_M)
            output_sizes: an int, output size of the fusion layer
            max_rank: an int, maximum rank for the CP decomposition
            eta: a float, hyperparameter for rank parameter distribution
            device:
            dtype:
        '''
        super().__init__()

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.max_rank = max_rank
        self.prior_type = prior_type
        self.eta = eta

        self.dropout = nn.Dropout(dropout)

        # initialize weight tensor factors
        factors = [nn.Parameter(torch.empty((input_size, max_rank), device=device, dtype=dtype)) \
            for input_size in input_sizes]
        factors = factors + [nn.Parameter(torch.empty((output_size, max_rank), device=device, dtype=dtype))]
        
        for factor in factors:
            nn.init.xavier_normal_(factor)

        self.weight_tensor_factors = nn.ParameterList(factors)

        # initialize rank parameter and its prior
        self.rank_parameter = nn.Parameter(torch.rand((max_rank,), device=device, dtype=dtype))

        if self.prior_type == 'half_cauchy':
            self.rank_parameter_prior_distribution = HalfCauchy(eta)
        elif self.prior_type == 'log_uniform':
            self.rank_parameter_prior_distribution = LogUniform(torch.tensor([1e-30], device=device, dtype=dtype), 
                                                                torch.tensor([1e30], device=device, dtype=dtype))

        # initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros((output_size,), device=device, dtype=dtype))
        else:
            self.bias = None
        
    def forward(self, inputs):

        # factorized forward propagation
        output = 1.0
        for x, factor in zip(inputs, self.weight_tensor_factors[:-1]):
            output = output * (x @ factor)

        output = output @ self.weight_tensor_factors[-1].T

        if self.bias is not None:
            output = output + self.bias

        output = F.relu(output)
        output = self.dropout(output)

        return output

    def get_log_prior(self):

        # clamp rank_param because <=0 is undefined 
        clamped_rank_parameter = self.rank_parameter.clamp(1e-30)
        self.rank_parameter.data = clamped_rank_parameter.data
        
        log_prior = torch.sum(self.rank_parameter_prior_distribution.log_prob(self.rank_parameter))
        
        # 0 mean normal distribution for the factors
        factor_prior_distribution = Normal(0, self.rank_parameter)
        for factor in self.weight_tensor_factors:
            log_prior = log_prior + torch.sum(factor_prior_distribution.log_prob(factor))
        
        return log_prior