# %%
# Setup for res_L1.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
import setup
import importlib
import seaborn as sns
import numpy as np

importlib.reload(setup)
from setup import (ResNet, all_ouptupt_featueres_contur_plots, list_saves, load, save, Trainer, HyperParameters, LogicGates, 
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

        
show_resnet_L1(trainer)

# %%


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
files = [f for f in files if 'lr0.001_s35500_' in f and 'T1000_D600_H600' in f]
for f in files:
    print(f)

# %%
for f in files:
    trainer = load(group, f)
    several_ouptupt_featueres_contur_plots(trainer, f)

# %%
for f in files:
    trainer = load(group, f)
    title = f'{f}, OR'
    print(title)
    all_ouptupt_featueres_contur_plots(trainer, gates='or', title=title)
# %%
for f in files:
    trainer = load(group, f)
    title = f'{f}, XOR'
    print(title)
    all_ouptupt_featueres_contur_plots(trainer, gates='xor', title=title)
# %%

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
z = torch.randn(5)

data = torch.stack((x, y, z))
corr_matrix = torch.corrcoef(data)
print(corr_matrix)
print("Correlation:", corr_matrix[0, 1])
# %%
data
# %%
files = list_saves(group)
files = [f for f in files if 'lr0.001_s35500_' in f and 'T1000_D600_H600' in f]


for f in files:
    print(f)
    trainer = load(group, f)

    #input = trainer.gates.generate_input_data(trainer.hp.bs*32, trainer.hp.nai)
    input = torch.eye(trainer.hp.T)
    target = trainer.gates.forward(input)
    active_relus = get_active_relus(trainer, input.T)[0].float()
    with torch.no_grad():
        output = trainer.model(input.T)

    corr = one_to_many_corr(target[-1], active_relus)

    plt.hist(corr.cpu().numpy(), bins=50)
    plt.ylim(0, 100)
    plt.show()

# %%
f = 'T1000_D600_H600_L1_nai5_tu0.1_bs4096_wd0.01_lr0.001_s35500_o1vnmzjm'
#input = torch.eye(trainer.hp.T)
input = trainer.gates.generate_input_data(trainer.hp.bs*32, trainer.hp.nai)
target = trainer.gates.forward(input)
active_relus = get_active_relus(trainer, input.T)[0].float()
with torch.no_grad():
    output = trainer.model(input.T)
# %%

corr = one_to_many_corr(target[-15], active_relus)

plt.hist(corr.cpu().numpy(), bins=100)
plt.ylim(0, 20)
plt.axvline(x=0.07, color='r', linestyle='dashed', linewidth=1)
plt.axvline(x=-0.07, color='r', linestyle='dashed', linewidth=1)
plt.show()
# %%

corr = many_to_many_corr(target.T, active_relus)


# %%
plt.figure(figsize=(8*5, 6*5))
sns.heatmap(corr[:100,:100].cpu().numpy(), cmap='bwr', center=0)
plt.title('Correlation Matrix of Active ReLUs')
plt.xlabel('Ouptput')
plt.ylabel('ReLU Neuron')
plt.show()
# %%

one = torch.tensor([1,1,0,0.])
many = torch.tensor([[1,1,0,0.], [1,0,1,0.], [0,0,1,1.]]).T
one_to_many_corr(one, many)
# %%


T = 80
D = 50
H = 100
gates = LogicGates(T, 'mixed')

for tu in [0, 0.01, 0.1]:
    for H in [100, 200]:
        hp = HyperParameters(T=T, D=D, H=H, L=1, nai=3, tu=tu,
                            bs=2048, lr=0.001, wd=0.01, ntype='resnet', gtype='mixed')
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
files = [f for f in files if 'T80_' in f and '_tu0_' in f and '_s35500_' in f]
for f in files:
    print(f)
    trainer = load(group, f)
    trainer.train(steps=15000, log_interval=100)
    save(trainer)
    trainer.train(steps=15000, log_interval=100)
    save(trainer)




# %%
