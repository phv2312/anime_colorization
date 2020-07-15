import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def _create_matrix(shape):
    return np.random.rand(*shape).astype(np.float32)

class AttentionModule(nn.Module):
    def __init__(self, in_dim):
        super(AttentionModule, self).__init__()
        self.in_dim = in_dim

        self.to_query = nn.Linear(in_features=self.in_dim, out_features=self.in_dim, bias=False)
        self.to_key   = nn.Linear(in_features=self.in_dim, out_features=self.in_dim, bias=False)
        self.to_value = nn.Linear(in_features=self.in_dim, out_features=self.in_dim, bias=False)

    def forward(self, query_input, key_input):
        """

        :param query_input: of size (batch, n_elem_q, dim)
        :param key_input: of size (batch, n_elem_k, dim)
        :return:
        """

        #
        key_output   = self.to_key(key_input)
        value_output = self.to_value(query_input)
        query_output = self.to_query(query_input)

        #
        a = torch.bmm(key_output, query_output.permute(dims=[0,2,1])) / np.sqrt(self.in_dim) #(batch, n_elem_k, n_elem_q)
        a = F.softmax(a, dim=-1) # (batch, n_elem_k, n_elem_q)

        #
        context = torch.bmm(a, value_output) # (batch, n_elem_k, dim)
        return context + key_input

