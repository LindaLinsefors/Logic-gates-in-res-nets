# %%
# Setup for and.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
import setup
import importlib
import seaborn as sns
import numpy as np

importlib.reload(setup)
from setup import (list_saves, load, save, Trainer, HyperParameters, LogicGates, 
                   _show_matrix, show_resnet, combine_linear, show_matrix,
                   ouptupt_featuere_contur_plot, several_ouptupt_featueres_contur_plots)

torch.set_default_device("cuda") # Set default device

wandb.login() # Login to wandb

group = 'Res L1'
# %%

T = 1000
gates = LogicGates(T, 'mixed')

nai = 5

gtype = 'mixed'
ntype = 'resnet'
L = 1

bs = 4096; wd=0.01; lr=0.001

f = 'T1000_D600_H1800_L1_nai5_tu0_bs4096_wd0.01_lr0.001_s35500_xgp3evh6'
trainer = load(group, f)
gates = trainer.gates

for D in [100, 200, 400, 600]:
    for H in [D, D*3]:
        for tu in [0.01]:
            hp = HyperParameters(T=T, D=D, H=H, L=L, nai=nai, tu=tu,
                                 bs=bs, lr=lr, wd=wd,
                                 gtype=gtype, ntype=ntype)
            trainer = Trainer(hp, group=group, gates=gates)
            trainer.train(steps=500, log_interval=10)
            save(trainer)
            trainer.train(steps=5000, log_interval=100)
            save(trainer)
            trainer.train(steps=15000, log_interval=100)
            save(trainer)
            trainer.train(steps=15000, log_interval=100)
            save(trainer)


# %%
files = list_saves(group)
for f in files:
    print(f)
# %%
files = [f for f in files if '_s35500_' in f]
for f in files:
    print(f)
# %%
trainer = load(group, 
    'T1000_D100_H100_L1_nai5_tu0.1_bs4096_wd0.01_lr0.01_s35500_ta7w10kf')
# %%
# Vissualize matrix weights
layer = trainer.model.input_layer
w = layer.weight.data.cpu().numpy()
b = layer.bias.data.cpu().numpy()
plt.figure(figsize=(8,6))
sns.heatmap(w, cmap='bwr', center=0)
plt.title('Weight Matrix Heatmap')
plt.xlabel('Input Neurons')
plt.ylabel('Output Neurons')
plt.show()
# %%

plt.hist(w.flatten(), bins=50)
plt.title('Weight Distribution')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.show()
# %%
net = trainer.model

in_w = net.input_layer.weight.data.cpu().numpy()
in_b = net.input_layer.bias.data.cpu().numpy()
out_w = net.output_layer.weight.data.cpu().numpy()
out_b = net.output_layer.bias.data.cpu().numpy()

h_in_w = net.hidden_layers[0].layer[0].weight.data.cpu().numpy()
h_in_b = net.hidden_layers[0].layer[0].bias.data.cpu().numpy()
h_out_w = net.hidden_layers[0].layer[2].weight.data.cpu().numpy()
h_out_b = net.hidden_layers[0].layer[2].bias.data.cpu().numpy()
# %%

show_matrix(in_w.T, title='Input Layer Weights', xlabel='Residual', ylabel='Input')
show_matrix(h_in_w, title='Hidden Layer Input Weights', xlabel='Residual', ylabel='Hidden') 
show_matrix(h_out_w.T, title='Hidden Layer Output Weights', xlabel='Residual', ylabel='Hidden')
show_matrix(out_w, title='Output Layer Weights', xlabel='Residual', ylabel='Output')
# %%

#Vissualise both weights and biases using heatmaps next to each other
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
sns.heatmap(in_w.T, cmap='bwr', center=0, ax=axs[0,0])
axs[0,0].set_title('Input Layer Weights')
axs[0,0].set_xlabel('Residual')
axs[0,0].set_ylabel('Input Neurons')
sns.heatmap(in_b.reshape(1, -1), cmap='bwr', center=0, ax=axs[0,1], cbar_kws={"shrink": 0.5})
axs[0,1].set_title('Input Layer Biases')    
axs[0,1].set_xlabel('Neuron')
axs[0,1].set_ylabel('Bias') 
sns.heatmap(out_w, cmap='bwr', center=0, ax=axs[1,0])
axs[1,0].set_title('Output Layer Weights')
axs[1,0].set_xlabel('Residual')
axs[1,0].set_ylabel('Output Neurons')
sns.heatmap(out_b.reshape(1, -1), cmap='bwr', center=0, ax=axs[1,1], cbar_kws={"shrink": 0.5})
axs[1,1].set_title('Output Layer Biases')
axs[1,1].set_xlabel('Neuron')
axs[1,1].set_ylabel('Bias')
plt.tight_layout()
plt.show()

# %%
show_matrix(in_b.reshape(1, -1), title='Input Layer Biases', xlabel='Residual', ylabel='Bias')
# %%


show_resnet(trainer)

# %%
plt.subplot(1,2,1)
_show_matrix(in_w.T, title='Input Layer Weights', xlabel='Residual', ylabel='Input')
plt.subplot(1,2,2)
_show_matrix(h_in_w, title='Hidden Layer Input Weights', xlabel='Residual', ylabel='Hidden') 
plt.show()
# %%
with torch.no_grad():

    net = trainer.model

    in_w = net.input_layer.weight.T
    in_w_norm = torch.norm(in_w, dim=1)
    in_b = net.input_layer.bias

    h_in_w = net.hidden_layers[0].layer[0].weight.T
    h_in_b = net.hidden_layers[0].layer[0].bias
    h_out_w = net.hidden_layers[0].layer[2].weight.T
    h_out_b = net.hidden_layers[0].layer[2].bias

    out_w = net.output_layer.weight.T
    out_b = net.output_layer.bias

    plt.plot(torch.norm(in_w, dim=1).cpu().numpy(), label='Input Layer Weights Norm')
    plt.show()


# %%
def show_resnet_L1(trainer):
    with torch.no_grad():
        direct = combine_linear(trainer.model.input_layer, trainer.model.output_layer)
        h_in = combine_linear(trainer.model.input_layer, trainer.model.hidden_layers[0].layer[0])
        h_out = combine_linear(trainer.model.hidden_layers[0].layer[2], trainer.model.output_layer)

        rows = 3
        cols = 2
        plt.figure(figsize=(8, rows * 3))

        plt.subplot(rows, cols, 1)
        d_w = direct.weight.data.cpu().numpy()
        _show_matrix(d_w, title='Direct Weights', xlabel='Input', ylabel='Output')
        plt.subplot(rows, cols, 2)
        d_b = direct.bias.data.cpu().numpy()
        _show_matrix(d_b.reshape(1, -1), title='Direct Biases', xlabel='Input', ylabel='Output')

        plt.subplot(rows, cols, 3)
        h_in_w = h_in.weight.data.cpu().numpy()
        _show_matrix(h_in_w, title='Hidden Layer Input Combined Weights', xlabel='Input', ylabel='Hidden')
        plt.subplot(rows, cols, 4)
        h_in_b = h_in.bias.data.cpu().numpy()
        _show_matrix(h_in_b.reshape(1, -1), title='Hidden Layer Input Combined Biases', xlabel='Input', ylabel='Hidden')
        plt.subplot(rows, cols, 5)
        h_out_w = h_out.weight.data.cpu().numpy()
        _show_matrix(h_out_w, title='Hidden Layer Output Combined Weights', xlabel='Hidden', ylabel='Output')
        plt.subplot(rows, cols, 6)
        h_out_b = h_out.bias.data.cpu().numpy()
        _show_matrix(h_out_b.reshape(1, -1), title='Hidden Layer Output Combined Biases', xlabel='Hidden', ylabel='Output') 
        plt.tight_layout()
        plt.show()
        
show_resnet_L1(trainer)

# %%
def show_direct(trainer, weights_only=False):
    with torch.no_grad():
        direct = combine_linear(trainer.model.input_layer, trainer.model.output_layer)

        if weights_only:
            plt.figure(figsize=(20, 20))
            d_w = direct.weight.data.cpu().numpy()
            _show_matrix(d_w, title='Direct Weights', xlabel='Input', ylabel='Output')
            plt.show()

        else:
            rows = 2
            cols = 1
            plt.figure(figsize=(8, 12))

            plt.subplot(rows, cols, 1)
            d_w = direct.weight.data.cpu().numpy()
            _show_matrix(d_w, title='Direct Weights', xlabel='Input', ylabel='Output')
            plt.subplot(rows, cols, 2)
            d_b = direct.bias.data.cpu().numpy()
            _show_matrix(d_b.reshape(1, -1), title='Direct Biases', xlabel='Input', ylabel='Output')

            plt.tight_layout()
            plt.show()

show_direct(trainer, weights_only=True)
# %%

min_val = -2.5
max_val = 2.5
direct = combine_linear(trainer.model.input_layer, trainer.model.output_layer)
clamped_weights = torch.clamp(direct.weight.data, min=min_val, max=max_val)

show_matrix(clamped_weights[trainer.gates.and_indices,].cpu().numpy(), 
            title='Direct Weights, AND', xlabel='Input', ylabel='Output')

show_matrix(clamped_weights[trainer.gates.or_indices,].cpu().numpy(), 
            title='Direct Weights, OR', xlabel='Input', ylabel='Output')

show_matrix(clamped_weights[trainer.gates.xor_indices,].cpu().numpy(), 
            title='Direct Weights, XOR', xlabel='Input', ylabel='Output')

# %%
min_val = -1
max_val = 1
direct = combine_linear(trainer.model.input_layer, trainer.model.output_layer)
clamped_weights = torch.clamp(direct.weight.data, min=min_val, max=max_val)

show_matrix(clamped_weights.cpu().numpy(), 
            title='Direct Weights, XOR', xlabel='Input', ylabel='Output')
# %%

second_inputs = trainer.gates.second_inputs
T = trainer.hp.T
plt.hist(second_inputs.cpu().numpy(), bins=np.arange(0.5, T + 1.5, 1))
plt.title('Second Inputs Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# %%

ouptput_feature = 0
input_features = [5,6]

x = torch.linspace(0, 1, 100)
y = torch.linspace(0, 1, 100)
X, Y = torch.meshgrid(x, y, indexing='ij')

inputs = torch.zeros(100, 100, T)
inputs[:, :, input_features[0]] = X
inputs[:, :, input_features[1]] = Y

with torch.no_grad():
    #outputs = trainer.model(inputs.reshape(-1, T))
    #Z = outputs[:, ouptput_feature].reshape(100, 100)

    outputs = trainer.model(inputs)
    Z = outputs[:, :, ouptput_feature]

plt.figure(figsize=(8, 6))
contour = plt.contourf(X.cpu(), Y.cpu(), Z.cpu(), levels=50, cmap='viridis')  # 'contourf' = filled contours
plt.colorbar(contour, label='f(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour Plot of f(x, y)')
plt.show()
# %%


ouptupt_featuere_contur_plot(trainer, 0)
# %%
several_ouptupt_featueres_contur_plots(trainer, 'hej')
# %%
trainer.name

# %%

files = list_saves(group)
files = [f for f in files if 'lr0.001_s35500_' in f and 'T1000_D600_H1800' in f]
for f in files:
    print(f)

# %%
for f in files:
    trainer = load(group, f)
    several_ouptupt_featueres_contur_plots(trainer, f)

# %%
