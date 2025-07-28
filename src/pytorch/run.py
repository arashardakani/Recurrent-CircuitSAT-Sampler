import glob
import logging
import os
import pathlib
import random

import numpy as np
import pandas as pd
from pysat.formula import CNF
from pysat.examples.genhard import PHP
from pysat.solvers import Solver
import gc
import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, KLDivLoss, BCEWithLogitsLoss, L1Loss
from tqdm import tqdm
from utils.seq_parser import *

import flags
import model.circuit_seq as circuit_seq

from utils.latency import timer


class Runner(object):
    def __init__(self, problem_type: str = "sat"):
        self.args = flags.parse_args()
        self.problem_type = problem_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        self.model = None
        self.loss = None
        self.optimizer = None
        self.module_name = None
        self.baseline = None
        self.save_dir = pathlib.Path(__file__).parent.parent / "results"
        self.datasets = []
        self.num_inputs = None
        self.dataset_str = ""
        self._setup_problems()

    def _setup_problems(self):
        """Setup the problem.
        """
        if self.args.problem_type == "BLIF":
            if self.args.dataset_path is None:
                raise NotImplementedError
            else:
                dataset_path = os.path.join(pathlib.Path(__file__).parent.parent, self.args.dataset_path)
                self.datasets = sorted(glob.glob(dataset_path))
                self.problems = None
                self.dataset_str = self.args.dataset_path.split('/')[-1].replace('.bench', '').replace('.', '')
            self.results={}
            logging.info(f"Dataset used: {self.dataset_str}")
        else:
            raise NotImplementedError

    def read_CircuitSAT_file(self, file_path):
        with open(file_path, 'r') as file:
            verilog_string = file.read()
        return verilog_string

    def _initialize_model(self, prob_id: int = 0):
        """Initialize problem-specifc model, loss, optimizer and input, target tensors
        for a given problem instance, e.g. a SAT problem (in CNF form).
        Note: must re-initialize the model for each problem instance.

        Args:
            prob_id (int, optional): Index of the problem to be solved. Defaults to 0.
        """
        problem = self.read_CircuitSAT_file(self.datasets[prob_id])
        self.solutions = {}
        if self.args.problem_type == "BLIF":
            if self.args.circuit_type == "seq":
                bench_code = problem.replace('.', '_')
                inputs, outputs, registers = parse_sequential_module(bench_code)
                num_outputs = len(outputs)
                module_name = self.datasets[prob_id].split('/')[-1].replace('.bench', '').replace('.', '')
                pytorch_model = generate_sequential_pytorch_model(module_name, inputs, outputs, registers, bench_code)
                self.num_inputs = len(inputs)
                self.num_outputs = num_outputs
        else:
            raise NotImplementedError

        if self.args.circuit_type == "seq":
            self.model = circuit_seq.CircuitModel(
                    pytorch_model=pytorch_model,
                    inputs_str = inputs,
                    num_clk_cycles = self.args.num_clock_cycles,
                    start_point = self.args.start_point,
                    module_name=module_name,
                    batch_size=self.args.batch_size,
                    device=self.device,
                )
        if self.args.problem_type == "BLIF":
            if self.args.circuit_type == "seq":
                if module_name in ['s27', 's2081', 's298', 's386', 's400', 's4201', 's444', 's510', 's526', 's8381', 's938', 's1423']:
                    self.target_idx = [-1]
                    self.target = torch.ones(self.args.batch_size,1, device = self.device)
                elif module_name in ['s298']:
                    self.target_idx = [-1]
                    self.target = torch.ones(self.args.batch_size,1, device = self.device)
                elif module_name in ['s635', 's526n', 's382']:
                    self.target_idx = [-1]
                    self.target = torch.zeros(self.args.batch_size,1, device = self.device)
                elif module_name in ['s344', 's349','s499', 's820', 's832', 's1238', 's1269', 's1488', 's3271','s3384', 's4863','s6669', 's13207']:
                    self.target_idx = [0, -1]
                    self.target = torch.ones(self.args.batch_size,2, device = self.device)
                    self.target[:, 0] = 0.
                elif module_name in ['s641', 's1196', 's713', 's953', 's967', 's991', 's1494', 's1512',  's5378', 's92341', 's9234', 's15850']:
                    self.target_idx = [0, 1]
                    self.target = torch.ones(self.args.batch_size,2, device = self.device)
                    self.target[:, 1] = 0.
                elif module_name in ['b01', 'b03', 'b04',  'b06', 'b08', 'b11', 'b13']:
                    self.target_idx = [-1]
                    self.target = torch.ones(self.args.batch_size,1, device = self.device)
                elif module_name in ['b02', 'b05', 'b07', 'b09']:
                    self.target_idx = [-1]
                    self.target = torch.zeros(self.args.batch_size,1, device = self.device)
                elif module_name in ['b10', 'b12']:
                    self.target_idx = [-2]
                    self.target = torch.zeros(self.args.batch_size,1, device = self.device)
                elif module_name in [ 'b18', 'b19']:
                    self.target_idx = [0, -1]
                    self.target = torch.ones(self.args.batch_size, 2, device = self.device)
                    self.target[:, -1] = 0.
                elif module_name in ['b18', 'b19', 'b20', 'b21', 'b22']:
                    self.target_idx = [0, 7, -1]
                    self.target = torch.ones(self.args.batch_size,3, device = self.device)
                    # target[:, 1:2] = 0.
                elif module_name in ['b14', 'b15', 'b17']:
                    self.target_idx = [0, -1]
                    self.target = torch.zeros(self.args.batch_size, 2, device = self.device)
                elif module_name in ['s35932']:
                    self.target_idx = [0, 15, 31, -1]
                    self.target = torch.ones(self.args.batch_size,4, device = self.device)
                    self.target[:, -1] = 0.
                else:
                    self.target_idx = [0, 15, 31, 63, -1]
                    self.target = torch.ones(self.args.batch_size,5, device = self.device)
                    self.target[:, 1:2] = 0.
            else:
                raise NotImplementedError


        self.module_name = module_name
        self.loss = MSELoss(reduction='sum')
        self.loss_per_batch = MSELoss(reduction='none') 
        
        
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.args.learning_rate
        )
        self.results[prob_id] = {
                "prob_desc": self.datasets[prob_id].split('/')[-1],
                "num_outputs": self.num_outputs,
                "num_inputs": self.num_inputs,
            }

        self.model.to(self.device)
        self.target.to(self.device)
        self.epochs_ran = 0



    def run_back_prop_seq(self, train_loop: range):
        """Run backpropagation for the given number of epochs."""
        
        for internal_loop_idx in range(self.args.start_point, self.args.num_clock_cycles):
            for epoch in train_loop:
                self.model.train()
                outputs_list = self.model(internal_loop_idx)
                output = torch.cat(outputs_list, dim = -1)[:,self.target_idx]
                loss = self.loss(output, self.target)
            
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            cnt = 0
            self.solutions[self.args.num_clock_cycles - 1 - internal_loop_idx] = {}
            for i in range(internal_loop_idx, -1, -1):
                sub_solutions = []
                for param in self.model.emb.parameters_list:
                    sub_solutions.append(param[:,self.args.num_clock_cycles - 1 - i].unsqueeze(-1).cpu())
                self.solutions[self.args.num_clock_cycles - 1 - internal_loop_idx][cnt] = sub_solutions
                cnt += 1
            self.model.emb.init(self.args.num_clock_cycles - 1 - internal_loop_idx, self.args.num_clock_cycles)
        return self.loss_per_batch(output, self.target)



    
    def _check_solution(self):
        sol_list = []
        

        if self.args.problem_type == "BLIF":
            if self.args.circuit_type == "seq":
                num_unique_solutions = 0
                with torch.no_grad():
                    self.model.probabilistic_circuit_model.init_registers()
                    self.model.probabilistic_circuit_model.set_registers(False)
                    idx = self.args.num_clock_cycles - 1
                    for internal_loop_idx in range(self.args.start_point, self.args.num_clock_cycles):
                        states = [state * 0. for state in self.model.probabilistic_circuit_model.call_registers()]
                        inputs = []
                        for j in range(len(self.solutions[idx - internal_loop_idx])):
                            inputs.append(torch.cat([(torch.sign(par.clone()) + 1)/2 for par in self.solutions[idx - internal_loop_idx][j]], dim = -1))
                            solutions_output, states = self.model.probabilistic_circuit_model([(torch.sign(par.to(self.device)) + 1)/2 for par in self.solutions[idx - internal_loop_idx][j]], states)
                            
                        _, rev_idxs = torch.cat(inputs, dim = -1).unique(dim=0, return_inverse=True)
                        first_occ_idxs = rev_idxs.unique()
                        final_output = torch.cat(solutions_output, dim = -1)[first_occ_idxs, :]
                        final_output = final_output[:, self.target_idx]
                        # print('Number of unique solutions:', torch.all(final_output == self.target[first_occ_idxs,:], dim = -1).sum(dim = -1))
                        sol_list.append([torch.all(final_output == self.target[first_occ_idxs,:], dim = -1).sum(dim = -1).item()])
                        num_unique_solutions += torch.all(final_output == self.target[first_occ_idxs,:], dim = -1).sum(dim = -1)
        print(sol_list)
        return num_unique_solutions

    
    
    def run_model(self, prob_id: int = 0):
        solutions_found = []
        if self.args.latency_experiment:
            train_loop = range(self.args.num_steps)
            if self.args.problem_type == "BLIF":
                if self.args.circuit_type == "seq":
                    elapsed_time, losses = timer(self.run_back_prop_seq)(train_loop)
            logging.info("--------------------")
            logging.info("NN model solving")
            logging.info(
                f"Elapsed Time: {elapsed_time:.6f} seconds"
            )
        else:
            train_loop = (
                range(self.args.num_steps)
                if self.args.verbose
                else tqdm(range(self.args.num_steps))
            )
            if self.args.problem_type == "BLIF":
                if self.args.circuit_type == "seq":
                    losses = self.run_back_prop_seq(train_loop)
        


        solutions_found = self._check_solution()

        self.results[prob_id].update(
            {
                "model_runtime": elapsed_time,
                "model_epochs_ran": self.args.num_steps,
            }
        )
        return solutions_found

    def run(self, prob_id: int = 0):
        """Run the experiment."""
        self._initialize_model(prob_id=prob_id)
        # run NN model solving
        solutions_found = self.run_model(prob_id)
        is_verified = solutions_found > 0
        self.results[prob_id].update(
            {
                "num_unique_solutions": solutions_found.long().cpu().tolist(),
            }
        )
        
        
        logging.info("--------------------\n")
        

    def run_all_with_baseline(self):
        """Run all the problems in the dataset given as argument to the Runner."""
        for prob_id in range(len(self.datasets)):
            self.run(prob_id=prob_id)
        if self.args.latency_experiment:
            self.export_results()

    def export_results(self):
        """Export results to a file."""
        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{self.problem_type}_{self.dataset_str}_{self.args.num_steps}"
        filename += f"_mse_{self.args.learning_rate}_{self.args.batch_size}.csv"
        filename = os.path.join(self.save_dir, filename)
        df = pd.DataFrame.from_dict(self.results)
        df = df.transpose()
        df.to_csv(filename, sep="\t", index=False)


if __name__ == "__main__":
    runner = Runner(problem_type="sat")
    runner.run_all_with_baseline()
