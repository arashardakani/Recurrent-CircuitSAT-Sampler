import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
import networkx as nx
import re






def extract_variables(expr, inputs, registers):
    # Regular expression pattern to match variables inside parentheses
    pattern = r'\((.*?)\)'
    matches = re.findall(pattern, expr)
    variables = [var.strip() for match in matches for var in match.split(',')]
    return variables

def refine(text, registers, inputs):
    lines = text.strip().split('\n')
    filtered_lines = [line for line in lines if not line.startswith('#') and 'INPUT' not in line and 'OUTPUT' not in line and 'DFF' not in line]
    registered_lines = [line for line in lines if 'DFF' in line]
    
    modified_lines = []
    states = []
    for line in registered_lines:
        variable, value = line.split('=')        
        modified_variable = f"{variable.strip()}"        
        value = value.split('(')[1].split(')')[0]
        states.append(value)
        modified_line = f"{modified_variable} = {value}"
        modified_lines.append(modified_line)
    
    filtered_string = '\n'.join(filtered_lines)
    non_empty_lines = [line for line in filtered_string.split('\n') if line.strip() != '']
    filtered_string = '\n'.join(non_empty_lines)


    dependencies = {}
    expressions = {}
    # Extract variable names and their dependencies
    for line in filtered_string.strip().split('\n'):
        parts = line.split('=')
        variable = parts[0].strip()
        expr = parts[1].strip() if len(parts) > 1 else ""
        dependencies[variable] = extract_variables(expr, inputs, registers)
        expressions[variable] = parts[1].strip()
    G = nx.DiGraph()
    for variable, deps in dependencies.items():
        for dep in deps:
            G.add_edge(dep, variable)
    # Topological sorting of statements
    sorted_vars = list(nx.topological_sort(G))
    rearranged_statements = [f"{var} = {expressions[var]}" for var in sorted_vars if var in expressions and var not in inputs and var not in registers]
    modified_input_string = '\n'.join(rearranged_statements)
    
    return modified_input_string.strip().split('\n'), states



def parse_sequential_module(verilog_code):
    lines = verilog_code.split('\n')
    inputs = []
    outputs = []
    registers = []
    
    for line in lines:
        
        if 'INPUT' in line:
            if line.startswith('INPUT('):
                # Extract the variable name between parentheses
                variable = line.split('(')[1].split(')')[0].replace('.', '')
                inputs.append(variable)
        elif 'OUTPUT' in line:
            if line.startswith('OUTPUT('):
                # Extract the variable name between parentheses
                variable = line.split('(')[1].split(')')[0].replace('.', '')
                outputs.append(variable)
        elif 'DFF' in line:
            variable = line.split('=')[0].strip().replace('.', '')
            registers.append(variable)
    return inputs, outputs, registers

def generate_sequential_pytorch_model(module_name, inputs, outputs, registers, assignments):
    class_name = module_name.lower()
    class_definition = f"class {class_name}(nn.Module):\n" \
                       f"    def __init__(self, batch_size, device):\n" \
                       f"        super().__init__()\n"
    class_definition += f"        self.batch_size = batch_size\n"
    class_definition += f"        self.device = device\n"
    for register in registers:
        class_definition += f"        self.{register} = nn.Parameter(torch.full((self.batch_size, 1), -3.5, device = self.device))\n"
    class_definition += f"    def init_registers(self, rand = False):\n"
    class_definition += f"        if rand:\n"
    for register in registers:
        class_definition += f"            nn.init.xavier_uniform_(self.{register}.data)\n"
    class_definition += f"        else:\n"
    for register in registers:
        class_definition += f"            self.{register}.data.fill_(-3.5)\n"
    class_definition += f"    def set_registers(self, set):\n"
    for register in registers:
        class_definition += f"        self.{register}.requires_grad = set\n"
    class_definition += f"    def call_registers(self):\n"
    for register in registers:
        class_definition += f"        self.{register}.data.clamp_(-3.5, 3.5)\n"
    class_definition += f"        return {', '.join(['sigmoid(20. * self.{})'.format(reg) for reg in registers])}\n"
    class_definition += "\n"
    class_definition += f"    def forward(self, inputs, registers):\n"
    
    class_definition += f"        {', '.join(inputs)+','} = inputs\n"
    for i, term in enumerate(registers, start=0):
        class_definition += f"        {term} = registers[{i}]\n"
    logics_assignments, registers_assignments = refine(assignments, registers, inputs)
    for assignment in logics_assignments:
        class_definition += f"        {assignment}\n"
    registers_assignments = [f"{assignment}" for assignment in registers_assignments]
    class_definition += f"        states = {', '.join(registers_assignments)}\n"
    class_definition += f"        outputs = [{', '.join(outputs)}]\n"
    class_definition += "\n" \
                       f"        return outputs, states\n"
    return class_definition










