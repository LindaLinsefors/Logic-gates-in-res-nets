# %%
# Setup

project_name = "Logic Gates in ResNets"

# Import necessary libraries
import os
import torch
import torch.nn as nn
import wandb
from line_profiler import LineProfiler, profile
import numpy as np
import matplotlib.pyplot as plt

# Set default device
torch.set_default_device("cuda")

wandb.login() # Login to wandb


class ResidualBlock(nn.Module):
    def __init__(self, D, H):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(D, H),
            nn.ReLU(),
            nn.Linear(H, D)
        )

    def forward(self, x):
        return x + self.layer(x)

# Residual Network with input and output dim T, and hidden dim D
class ResNet(nn.Module):
    def __init__(self, T, D, H, L):
        super(ResNet, self).__init__()
        self.input_layer = nn.Linear(T, D)
        self.hidden_layers = nn.ModuleList([ResidualBlock(D, H) for _ in range(L)])
        self.output_layer = nn.Linear(D, T)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
    
    def active_relu_count(self, x):
        with torch.no_grad():
            x = self.input_layer(x)
            count = []
            for layer in self.hidden_layers:
                pre_activation = layer.layer[0](x)
                count.append((pre_activation > 0).sum().item())
                x = layer(x)
            return count
    

class LogicGates(object):
    def __init__(self, T):
        self.T = T

        self.first_inputs = torch.arange(T)
        self.second_inputs = (self.first_inputs + torch.randint(1, T-1, (T,))) % T

        self.connections = torch.eye(T, dtype=torch.int32) #Add first input connections
        self.connections[torch.arange(T), self.second_inputs] = 1 #Add second input connections

        self.and_indices = torch.arange(T//3)
        self.or_indices = torch.arange(T//3, 2*T//3)
        self.xor_indices = torch.arange(2*T//3, T)

        self.connections_cuda = self.connections.float()

    def forward(self, x):
        x = self.connections @ x
        output = torch.zeros_like(x, dtype=torch.int32)
        output[self.and_indices] = (x[self.and_indices]==2).type(torch.int32)
        output[self.or_indices] = (x[self.or_indices] >= 1).type(torch.int32)
        output[self.xor_indices] = (x[self.xor_indices]==1).type(torch.int32)
        return output
    
    def forward_cuda(self, x):
        x = self.connections_cuda @ x
        output = torch.zeros_like(x)
        output[self.and_indices] = (x[self.and_indices]==2).float()
        output[self.or_indices] = (x[self.or_indices] >= 1).float()
        output[self.xor_indices] = (x[self.xor_indices]==1).float()
        return output

    def generate_input_data(self, batch_size):
        inputs = torch.zeros((self.T, batch_size), dtype=torch.int32)

        active_and_outputs = torch.randint(self.T//3, (batch_size,))
        one_more_active_input = torch.randint(self.T, (batch_size,))

        batch_range = torch.arange(batch_size)

        inputs[active_and_outputs, batch_range] = 1
        #inputs[self.first_inputs[active_and_outputs], batch_range] = 1 #Same as previous row
        inputs[self.second_inputs[active_and_outputs], batch_range] = 1
        inputs[one_more_active_input, batch_range] = 1

        return inputs
    

    def generate_input_data_cuda(self, batch_size):
        inputs = torch.zeros((self.T, batch_size))

        active_and_outputs = torch.randint(self.T//3, (batch_size,))
        one_more_active_input = torch.randint(self.T, (batch_size,))

        batch_range = torch.arange(batch_size)

        inputs[active_and_outputs, batch_range] = 1
        #inputs[self.first_inputs[active_and_outputs], batch_range] = 1 #Same as previous row
        inputs[self.second_inputs[active_and_outputs], batch_range] = 1
        inputs[one_more_active_input, batch_range] = 1

        return inputs

class HyperParameters:
    def __init__(self, T, D, H, L, tu=0, bs=2048, lr=1e-3):
        self.T = T  # Number of input and output features
        self.D = D  # Hidden dimension
        self.H = H  # Hidden layer dimension
        self.L = L  # Number of residual blocks
        self.tu = tu  # Target uncertainty
        self.bs = bs  # Batch size
        self.lr = lr  # Learning rate



        

class Trainer:
    def __init__(self, hp, group='Test', name=None, project=project_name):
        self.hp = hp # Hyperparameters
        self.model = ResNet(hp.T, hp.D, hp.H, hp.L)
        self.gates = LogicGates(hp.T)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hp.lr)
        self.current_step = 0

        self.group = group  # Experiment group
        if name is None:
            self.name = f"T{hp.T}_D{hp.D}_H{hp.H}_L{hp.L}_tu{hp.tu}_bs{hp.bs}_lr{hp.lr}"
        else:
            self.name = name

        with wandb.init(project=project, group=self.group, name=self.name, 
                        config=vars(self.hp),
                        settings=wandb.Settings(silent=True)) as run:
            self.run_id = run.id



    def train(self, steps, log_interval=100):

        with wandb.init(project=project_name, id=self.run_id, resume="must",
                        settings=wandb.Settings(silent=True)):

            for _ in range(steps):
                inputs = self.gates.generate_input_data_cuda(self.hp.bs)
                targets = self.gates.forward_cuda(inputs).float()
                
                inputs = inputs.float()
                if self.hp.tu != 0:
                    targets = targets * (1 - 2*self.hp.tu) + self.hp.tu

                self.optimizer.zero_grad()
                outputs = self.model(inputs.T)
                loss = self.criterion(outputs, targets.T)
                loss.backward()
                self.optimizer.step()

                if self.current_step % log_interval == 0:
                    log_data = {"loss": loss.item()}
                    log_data['average_absolute_logits'] = outputs.abs().mean().item()

                    active_relu_count = self.model.active_relu_count(inputs.T)
                    for l in range(self.hp.L):
                        log_data[f"layer_{l+1}_active_relu_count"] = active_relu_count[l]/self.hp.bs

                    wandb.log(log_data, step=self.current_step)
                
                self.current_step += 1

        return loss.item()
    

def save(trainer):
    os.makedirs(trainer.group, exist_ok=True)
    file_name = f"{trainer.name}_s{trainer.current_step}_{trainer.run_id}"
    path = os.path.join(trainer.group, file_name)

    torch.save(trainer, path)
    print(f"Model saved to {path}")

    return trainer.group, file_name

def list_saves(group):
    files = os.listdir(group)
    return files

def load(group, file_name):
    path = os.path.join(group, file_name)
    trainer = torch.load(path)
    print(f"Model loaded from {path}")
    return trainer



# %%
# Test

test_hp = HyperParameters(T=16, D=8, H=16, L=2, bs=2048, lr=1e-3)
trainer_s = Trainer(test_hp, group='Test')
trainer_s.train(steps=500, log_interval=10)
group, file_name = save(trainer_s)
trainer_l = load(group, file_name)
trainer_l.train(steps=5000, log_interval=100)
save(trainer_l)

# %%
test_hp = HyperParameters(T=128, D=64, H=64, L=2, bs=512, lr=1e-3)
trainer = Trainer(test_hp, group='Test')
trainer.train_step(steps=500, log_interval=10)
trainer.train_step(steps=5000, log_interval=100)

# %%
trainer.train_step(steps=5000, log_interval=100)
# %%

group = 'With target uncertainty'
hp = HyperParameters(T=6, D=6, H=6, L=2, tu=0.1, bs=2048, lr=1e-3)
trainer = Trainer(hp, group=group)
trainer.train(steps=500, log_interval=10)
save(trainer)
trainer.train(steps=5000, log_interval=100)
save(trainer)
# %%
trainer.train(steps=5000, log_interval=100)
# %%

trainer.run_id


# %%
group = 'With target uncertainty'
for _ in range(3):
    for tu in [0, 0.01, 0.1]:
        hp = HyperParameters(T=6, D=6, H=6, L=2, tu=tu, bs=2048, lr=1e-3)
        trainer = Trainer(hp, group=group)
        trainer.train(steps=500, log_interval=10)
        save(trainer)
        trainer.train(steps=500, log_interval=20)
        save(trainer)
        trainer.train(steps=5000, log_interval=50)
        save(trainer)
        trainer.train(steps=5000, log_interval=100)
        save(trainer)
# %%


files = list_saves(group)
files_s11000 = [f for f in files if '_s11000_' in f]
for f in files_s11000:
    print(f)

# %%

trainers_tu = {}
trainers_tu[0] = [load(group, f) for f in files_s11000 if '_tu0_' in f]
trainers_tu[0.01] = [load(group, f) for f in files_s11000 if '_tu0.01_' in f]
trainers_tu[0.1] = [load(group, f) for f in files_s11000 if '_tu0.1_' in f] 
# %%
net = trainers_tu[0][0].model
net.parameters
# %%
w_in = net.input_layer.weight.data
w_1_in = net.hidden_layers[0].layer[0].weight.data
w_1_out = net.hidden_layers[0].layer[2].weight.data
w_2_in = net.hidden_layers[1].layer[0].weight.data
w_2_out = net.hidden_layers[1].layer[2].weight.data
w_out = net.output_layer.weight.data

b_in = net.input_layer.bias.data
b_1_in = net.hidden_layers[0].layer[0].bias.data
b_1_out = net.hidden_layers[0].layer[2].bias.data
b_2_in = net.hidden_layers[1].layer[0].bias.data
b_2_out = net.hidden_layers[1].layer[2].bias.data
b_out = net.output_layer.bias.data


