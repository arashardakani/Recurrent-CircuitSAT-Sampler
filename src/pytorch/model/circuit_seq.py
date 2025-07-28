import argparse
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import gc
from .pgates import *
from torch.nn.functional import sigmoid




class PIEmbedding(nn.Module):
    def __init__(
        self,
        input_names: list[str],
        input_shape: list[int],
        device: str = "cpu",
        batch_size: int = 1,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.device = device
        self.batch_size = batch_size
        self.input_names = input_names

        self.parameters_list = nn.ParameterList()
        for name, size in zip(input_names, input_shape):
            param = nn.Parameter(torch.randn(batch_size, size, device=device))
            self.parameters_list.append(param)

                
        self.activation = torch.nn.Sigmoid()  
    def init(self, lower_idx, upper_idx):
        for param in self.parameters_list:
            selected_rows = param[:,lower_idx:upper_idx]
            nn.init.xavier_uniform_(selected_rows)
            param.data[:,lower_idx:upper_idx] = selected_rows


    def forward(self, idx):
        outputs = []
        for param in self.parameters_list:
            param[:,idx].data.clamp_(-3.5, 3.5)
            output_tensor = self.activation(2 * param[:,idx].unsqueeze(-1))
            outputs.append(output_tensor)
        return outputs
        





class CircuitModel(nn.Module):
    """Combinational Circuit instantiated from a PySAT CNF problem"""

    def __init__(self, **kwargs):
        # read cnf file
        super().__init__()
        
        self.pytorch_model = kwargs["pytorch_model"]
        exec(self.pytorch_model)
        class_object = locals()[kwargs["module_name"]]
        self.num_clk_cycles = kwargs["num_clk_cycles"]
        self.start_point = kwargs["start_point"]

        
        self.emb = PIEmbedding(kwargs["inputs_str"], [self.num_clk_cycles] * len(kwargs["inputs_str"]), kwargs["device"], kwargs["batch_size"])
        self.probabilistic_circuit_model = class_object(kwargs["batch_size"], kwargs["device"])
        self.probabilistic_circuit_model.init_registers()
        self.probabilistic_circuit_model.set_registers(False)

    def forward(self, internal_loop_idx):
        states = self.probabilistic_circuit_model.call_registers()
        for i in range(internal_loop_idx, -1, -1):
            x = self.emb(self.num_clk_cycles - 1 - i)
            out, states = self.probabilistic_circuit_model(x, states)
        return out


