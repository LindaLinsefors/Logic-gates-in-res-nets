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


torch.set_default_device("cuda") # Set default device

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
        
class MLP(nn.Module):
    def __init__(self, T, D, L):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(T, D)
        self.hidden_layers = nn.ModuleList([nn.Linear(D, D) for _ in range(L-1)])
        self.output_layer = nn.Linear(D, T)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x
    
    def active_relu_count(self, x):
        with torch.no_grad():
            count = []
            x = self.input_layer(x)
            count.append((x > 0).sum().item())
            x = torch.relu(x)
            for layer in self.hidden_layers:
                x = layer(x)
                count.append((x > 0).sum().item())
                x = torch.relu(x)
            return count

    
class LogicGates(object):
    def __init__(self, T, gtype):
        self.T = T
        self.gtype = gtype

        self.first_inputs = torch.arange(T)
        self.second_inputs = (self.first_inputs + torch.randint(1, T-1, (T,))) % T

        self.connections = torch.eye(T) #Add first input connections
        self.connections[torch.arange(T), self.second_inputs] = 1 #Add second input connections

        if gtype == 'mixed':
            self.and_indices = torch.arange(T//3)
            self.or_indices = torch.arange(T//3, 2*T//3)
            self.xor_indices = torch.arange(2*T//3, T)
        elif gtype == 'and':
            self.and_indices = torch.arange(T)
            self.or_indices = torch.tensor([], dtype=torch.int64)
            self.xor_indices = torch.tensor([], dtype=torch.int64)

        self.number_of_and = len(self.and_indices)
    
    def forward(self, x):
        x = self.connections @ x
        output = torch.zeros_like(x)
        output[self.and_indices] = (x[self.and_indices]==2).float()
        output[self.or_indices] = (x[self.or_indices] >= 1).float()
        output[self.xor_indices] = (x[self.xor_indices]==1).float()
        return output

    
    def generate_input_data(self, batch_size, num_active_inputs):
        inputs = torch.zeros((self.T, batch_size))
        batch_range = torch.arange(batch_size)

        for _ in range(num_active_inputs//2):
            active_pairs = torch.randint(self.T, (batch_size,))

            inputs[active_pairs, batch_range] = 1
            #inputs[self.first_inputs[active_pairs], batch_range] = 1 #Same as previous row
            inputs[self.second_inputs[active_pairs], batch_range] = 1

        if num_active_inputs % 2 == 1:
            extra_active = torch.randint(self.T, (batch_size,))
            inputs[extra_active, batch_range] = 1

        return inputs

class HyperParameters:
    def __init__(self, T, D, H, L, 
                 nai=3, tu=0, 
                 bs=2048, lr=1e-3, wd=0,
                 ntype='resnet', gtype='mixed'):
        
        self.T = T  # Number of input and output features
        self.D = D  # Hidden dimension
        self.H = H  # Hidden layer dimension
        self.L = L  # Number of residual blocks

        self.nai = nai  # Number of active inputs
        self.tu = tu  # Target uncertainty

        self.bs = bs  # Batch size
        self.lr = lr  # Learning rate
        self.wd = wd  # Weight decay

        self.ntype = ntype  # Net type, 'resnet' or 'mlp'
        self.gtype = gtype  # Logic gate type, 'mixed' or 'and'


        

class Trainer:
    def __init__(self, hp, group='Test', name=None, project=project_name, gates=None):
        self.hp = hp # Hyperparameters

        if hp.ntype == 'mlp':
            self.model = MLP(hp.T, hp.D, hp.L)
        elif hp.ntype == 'resnet':
            self.model = ResNet(hp.T, hp.D, hp.H, hp.L)

        if gates is None:
            self.gates = LogicGates(hp.T, hp.gtype)
        else:
            self.gates = gates

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=hp.lr, weight_decay=hp.wd)
        #self.optimiser = torch.optim.Adam(self.model.parameters(), lr=hp.lr)
        self.current_step = 0

        self.group = group  # Experiment group
        if name is None:
            if hp.H is None:
                self.name = f"T{hp.T}_D{hp.D}_L{hp.L}_nai{hp.nai}_tu{hp.tu}_bs{hp.bs}_wd{hp.wd}_lr{hp.lr}"
            else:
                self.name = f"T{hp.T}_D{hp.D}_H{hp.H}_L{hp.L}_nai{hp.nai}_tu{hp.tu}_bs{hp.bs}_wd{hp.wd}_lr{hp.lr}"
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
                inputs = self.gates.generate_input_data(self.hp.bs, self.hp.nai)
                targets = self.gates.forward(inputs).float()

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

def positive_bias_mask(layer):
    b = layer.bias.data
    return (b > 0)

def plot_layer(layer):
    w = layer.weight.data
    b = layer.bias.data
    nw = w/(b.abs())[:,None]
    positive_bias = positive_bias_mask(layer)

    plt.figure(figsize=(8,12))

    plt.subplot(3,2,1)
    plt.hist(w[positive_bias].cpu().numpy().flatten(), bins=50)
    plt.title('Input weights for neurons with positive bias')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')

    plt.subplot(3,2,2)
    plt.hist(w[~positive_bias].cpu().numpy().flatten(), bins=50)
    plt.title('Input weights for neurons with negative bias')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')

    plt.subplot(3,2,3)
    plt.hist(b[positive_bias].cpu().numpy().flatten(), bins=50)
    plt.title('Biases for neurons with positive bias')
    plt.xlabel('Bias Value')
    plt.ylabel('Count')

    plt.subplot(3,2,4)
    plt.hist(b[~positive_bias].cpu().numpy().flatten(), bins=50)
    plt.title('Biases for neurons with negative bias')
    plt.xlabel('Bias Value')
    plt.ylabel('Count')

    plt.subplot(3,2,5)
    plt.hist(nw[positive_bias].cpu().numpy().flatten(), bins=50)
    plt.title('Normalized Weights for neurons with positive bias')
    plt.xlabel('Normalized Weight Value')
    plt.ylabel('Count') 

    plt.subplot(3,2,6)
    plt.hist(nw[~positive_bias].cpu().numpy().flatten(), bins=50)
    plt.title('Normalized Weights for neurons with negative bias')
    plt.xlabel('Normalized Weight Value')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()


def _show_matrix(matrix, title='Matrix Heatmap', xlabel='Input', ylabel='Output'):
    sns.heatmap(matrix, cmap='bwr', center=0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

import seaborn as sns
def show_matrix(matrix, title='Matrix Heatmap', xlabel='Input', ylabel='Output'):
    _show_matrix(matrix, title, xlabel, ylabel)
    plt.show()

# %%
def show_resnet(trainer):
    net = trainer.model
    L = trainer.hp.L

    rows = (L + 1) * 2
    cols = 2

    plt.figure(figsize=(8, rows * 3))

    plt.subplot(rows, cols, 1)
    in_w = net.input_layer.weight.data.cpu().numpy()
    _show_matrix(in_w.T, title='Input Layer Weights', xlabel='Residual', ylabel='Input')
    plt.subplot(rows, cols, 2)
    in_b = net.input_layer.bias.data.cpu().numpy()
    _show_matrix(in_b.reshape(1, -1), title='Input Layer Biases', xlabel='Residual', ylabel='Bias')

    for l in range(L):
        plt.subplot(rows, cols, 2*(l+1)+1)
        h_in_w = net.hidden_layers[l].layer[0].weight.data.cpu().numpy()
        _show_matrix(h_in_w, title=f'Residual Block {l+1} Input Weights', xlabel='Residual', ylabel='Hidden') 
        plt.subplot(rows, cols, 2*(l+1)+2)
        h_in_b = net.hidden_layers[l].layer[0].bias.data.cpu().numpy()
        _show_matrix(h_in_b.reshape(-1, 1), title=f'Residual Block {l+1} Input Biases', xlabel='Bias', ylabel='Hidden')

        plt.subplot(rows, cols, 2*(L + l +1)+1)
        h_out_w = net.hidden_layers[l].layer[2].weight.data.cpu().numpy()
        _show_matrix(h_out_w.T, title=f'Residual Block {l+1} Output Weights', xlabel='Residual', ylabel='Hidden')
        plt.subplot(rows, cols, 2*(L + l +1)+2)
        h_out_b = net.hidden_layers[l].layer[2].bias.data.cpu().numpy()
        _show_matrix(h_out_b.reshape(1, -1), title=f'Residual Block {l+1} Output Biases', xlabel='Residual', ylabel='Bias')

    plt.subplot(rows, cols, rows*cols -1)
    out_w = net.output_layer.weight.data.cpu().numpy()
    _show_matrix(out_w, title='Output Layer Weights', xlabel='Residual', ylabel='Output')
    plt.subplot(rows, cols, rows*cols)
    out_b = net.output_layer.bias.data.cpu().numpy()
    _show_matrix(out_b.reshape(-1, 1), title='Output Layer Biases', xlabel='Bias', ylabel='Output')

    plt.tight_layout()
    plt.show()

def combine_linear(layer1, layer2):
    combined_layer = nn.Linear(layer1.in_features, layer2.out_features)
    combined_layer.weight.data = layer2.weight.data @ layer1.weight.data
    combined_layer.bias.data = layer2.weight.data @ layer1.bias.data + layer2.bias.data
    return combined_layer



def _ouptupt_featuere_contur_plot(trainer, ouptput_feature, text=True):

    input_features = [trainer.gates.first_inputs[ouptput_feature].item(), 
                      trainer.gates.second_inputs[ouptput_feature].item()]
    
    x = torch.linspace(0, 1, 100)
    y = torch.linspace(0, 1, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    inputs = torch.zeros(100, 100, trainer.hp.T)
    inputs[:, :, input_features[0]] = X
    inputs[:, :, input_features[1]] = Y

    with torch.no_grad():
        #outputs = trainer.model(inputs.reshape(-1, T))
        #Z = outputs[:, ouptput_feature].reshape(100, 100)

        outputs = trainer.model(inputs)
        Z = outputs[:, :, ouptput_feature]

    
    contour = plt.contourf(X.cpu(), Y.cpu(), Z.cpu(), levels=50, cmap='viridis') 

    plt.colorbar(contour)
    if text:
        plt.xlabel('First input')
        plt.ylabel('Second input')
        plt.title(f'Logits for output feature {ouptput_feature}')
    

def ouptupt_featuere_contur_plot(trainer, ouptput_feature):
    plt.figure(figsize=(8, 6))
    _ouptupt_featuere_contur_plot(trainer, ouptput_feature)
    plt.show()


def several_ouptupt_featueres_contur_plots(trainer, title=None):
    rows = 3
    cols = 4
    fig = plt.figure(figsize=(cols * 5, rows * 4 + 0.5))

    if title is not None:
        fig.suptitle(title, fontsize=16)
        #plt.subplots_adjust(top=0.9)

    for i in range(rows):
        for j in range(cols):
            plt.subplot(rows, cols, i * cols + j + 1)
            
            if i==0:
                output_feature = trainer.gates.and_indices[j]
            elif i==1:
                output_feature = trainer.gates.or_indices[j]
            else:
                output_feature = trainer.gates.xor_indices[j]

            _ouptupt_featuere_contur_plot(trainer, output_feature)
    plt.tight_layout()
    plt.show()

def all_ouptupt_featueres_contur_plots(trainer, gates='all', title=None):
    if gates == 'and':
        features = trainer.gates.and_indices
    elif gates == 'or':
        features = trainer.gates.or_indices
    elif gates == 'xor':
        features = trainer.gates.xor_indices
    else:
        features = torch.arange(trainer.hp.T)

    n_features = len(features)
    cols = 6
    rows = (n_features + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 5, rows * 4 + 0.5))

    if title is not None:
        fig.suptitle(title, fontsize=16)
        #plt.subplots_adjust(top=0.9)

    for idx, output_feature in enumerate(features):
        plt.subplot(rows, cols, idx + 1)
        _ouptupt_featuere_contur_plot(trainer, output_feature.item(), text=False)
    
    plt.tight_layout()
    plt.show()
  

# %%
