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
from setup import list_saves, load, save, Trainer, HyperParameters

torch.set_default_device("cuda") # Set default device

wandb.login() # Login to wandb

group = 'And MLP v2'
#Fixes from v1:
# - active_and_outputs are drawn from all and gates, not just T//3

# %%
group = 'And MLP'

hp = HyperParameters(T=1000, D=100, H=None, L=1, 
                    tu=0, bs=1024//2, lr=3e-2,
                    ntype = 'mlp', gtype = 'and')

trainer = Trainer(hp, group=group)
trainer.train(steps=500, log_interval=10)
save(trainer)
trainer.train(steps=5000, log_interval=100)
save(trainer)
trainer.train(steps=5000, log_interval=100)
save(trainer)
trainer.train(steps=5000, log_interval=100)
save(trainer)
trainer.train(steps=15000, log_interval=100)
save(trainer)



# %%
for tu in [0.01, 0.1]:
    for lr in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
        for bs in [256, 512, 1024, 2048]:
            hp = HyperParameters(T=1000, D=100, H=None, L=1, 
                                tu=tu, bs=bs, lr=lr,
                                ntype = 'mlp', gtype = 'and')

            trainer = Trainer(hp, group=group)
            trainer.train(steps=500, log_interval=10)
            save(trainer)
            trainer.train(steps=5000, log_interval=100)
            save(trainer)
            trainer.train(steps=15000, log_interval=100)
            save(trainer)
# %% ####################################################################
# Testing how the loss works


inputs = trainer.gates.generate_input_data_cuda(trainer.hp.bs)
targets = trainer.gates.forward_cuda(inputs).float()
inputs = inputs.float()

if trainer.hp.tu != 0:
    targets = targets * (1 - 2*trainer.hp.tu) + trainer.hp.tu

loss = nn.BCELoss()(targets, targets)

print(loss)
# %%

tu = torch.tensor(trainer.hp.tu)
loss = -tu*torch.log(tu) + (1 - tu)*torch.log(1 - tu)
print(loss)

# %% #####################################################################
# Investigating drop in layer_1_active_relu_count for T1000_D100_L1_tu0.1_bs256_lr0.03

files = list_saves(group)
for f in files:
    if 'T1000_D100_L1_tu0.1_bs256_lr0.03' in f:
        print(f)

# %%
run_id = 'yxwjp7rn'
name = 'T1000_D100_L1_tu0.1_bs256_lr0.03'
early_name = name + '_s5500_' + run_id
late_name = name + '_s20500_' + run_id

early = load(group, early_name)
late = load(group, late_name)
# %%

bs = 1024*4
inputs = early.gates.generate_input_data_cuda(bs).float()
targets = early.gates.forward_cuda(inputs).float()
if early.hp.tu != 0:
    targets = targets * (1 - 2*early.hp.tu) + early.hp.tu

with torch.no_grad():
    early_outputs = early.model(inputs.T)
    late_outputs = late.model(inputs.T)
    early_probs = torch.sigmoid(early_outputs)
    late_probs = torch.sigmoid(late_outputs)

active = targets.T > 0.5

bins = torch.linspace(0, 1, steps=150).cpu()

plt.hist(early_probs[active].cpu().numpy(), bins=bins, alpha=0.5, label='Early Active', color='blue', density=True)
plt.hist(late_probs[active].cpu().numpy(), bins=bins, alpha=0.5, label='Late Active', color='cyan', density=True)
plt.hist(early_probs[~active].cpu().numpy(), bins=bins, alpha=0.5, label='Early Inactive', color='red', density=True)
plt.hist(late_probs[~active].cpu().numpy(), bins=bins, alpha=0.5, label='Late Inactive', color='orange', density=True)
plt.axvline(0.9, color='black', linestyle='dashed')
plt.axvline(0.1, color='black', linestyle='dashed')
plt.legend()
plt.title('Output Probability Distributions')
plt.show()

bins = torch.linspace(-5, 5, steps=150).cpu()

plt.hist(early_outputs[active].cpu().numpy(), bins=bins, alpha=0.5, label='Early Active', color='blue', density=True)
plt.hist(late_outputs[active].cpu().numpy(), bins=bins, alpha=0.5, label='Late Active', color='cyan', density=True)
plt.hist(early_outputs[~active].cpu().numpy(), bins=bins, alpha=0.5, label='Early Inactive', color='red', density=True)
plt.hist(late_outputs[~active].cpu().numpy(), bins=bins, alpha=0.5, label='Late Inactive', color='orange', density=True)
plt.axvline(torch.logit(torch.tensor(0.9)).cpu(), color='black', linestyle='dashed')
plt.axvline(torch.logit(torch.tensor(0.1)).cpu(), color='black', linestyle='dashed')
plt.legend()
plt.title('Output Distributions')
plt.show()

# %%
eps = 0.05
bins = torch.linspace(0.1 - eps, 0.1 + eps, steps=150).cpu()

plt.hist(early_probs[~active].cpu().numpy(), bins=bins, alpha=0.5, label='Early Inactive', color='red', density=True)
plt.hist(late_probs[~active].cpu().numpy(), bins=bins, alpha=0.5, label='Late Inactive', color='orange', density=True)
plt.legend()
plt.title('Output Probability Distributions')
plt.show()


# %%

bins = torch.linspace(0, 1, steps=150).cpu()
with torch.no_grad():
    early_active_neurons = early.model.input_layer.forward(inputs.T) > 0
    late_active_neurons = late.model.input_layer.forward(inputs.T) > 0

    early_neuron_activation_frequency = early_active_neurons.sum(dim=0) / bs
    late_neuron_activation_frequency = late_active_neurons.sum(dim=0) / bs

    plt.hist(early_neuron_activation_frequency.cpu().numpy(), bins=bins, alpha=0.5, label='Early', color='blue', density=True)
    plt.hist(late_neuron_activation_frequency.cpu().numpy(), bins=bins, alpha=0.5, label='Late', color='cyan', density=True)
    plt.legend()
    plt.title('Neuron Activation Frequency')
    plt.show()


# %%
print(f"Early Neuron Activation Frequency Mean: {early_neuron_activation_frequency.mean()}")
print(f"Late Neuron Activation Frequency Mean: {late_neuron_activation_frequency.mean()}")  
# %%

for bs in [1024, 2048, 4096, 8192]:
    print(f"Batch Size: {bs}")
    inputs = early.gates.generate_input_data_cuda(bs).float()
    print(early.model.active_relu_count(inputs.T)[0]/bs)
    print(late.model.active_relu_count(inputs.T)[0]/bs)
# %%

plt.hist(late_neuron_activation_frequency.cpu().numpy(), bins=100, alpha=0.5, label='Late', color='cyan', density=True)
plt.legend()
plt.title('Neuron Activation Frequency')
plt.show()
# %%


plt.plot(late.model.input_layer.bias.data.cpu(), late_neuron_activation_frequency.cpu(), 'o')
plt.title('Neuron Activation Frequency vs Biases')
plt.ylabel('Activation Frequency')
plt.xlabel('Bias Value')
plt.show()


# %%
# visualizing weight using a heat map

trainer = early

w1 = trainer.model.input_layer.weight.data.cpu()
b1 = trainer.model.input_layer.bias.data.cpu()
w2 = trainer.model.output_layer.weight.data.cpu()
b2 = trainer.model.output_layer.bias.data.cpu()

nw1 = w1 / (-b1)[:, None]
nw2 = w2 * (-b1)[None, :]

import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(w1, cmap='viridis')
plt.title('Weight Heatmap, Input')
plt.xlabel('Input Neuron')
plt.ylabel('Hidden Neuron')
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(nw1, cmap='viridis')
plt.title('Normalized Weight Heatmap, Input')
plt.xlabel('Input Neuron')
plt.ylabel('Hidden Neuron')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(w2, cmap='viridis')
plt.title('Weight Heatmap, Output')
plt.xlabel('Hidden Neuron')
plt.ylabel('Output Neuron')
plt.show() 

plt.figure(figsize=(10, 8))
sns.heatmap(nw2, cmap='viridis')
plt.title('Normalized Weight Heatmap, Output')
plt.xlabel('Hidden Neuron')
plt.ylabel('Output Neuron')
plt.show()

# %%
plt.figure(figsize=(10, 20))
sns.heatmap(nw2, cmap='viridis')
plt.title('Normalized Weight Heatmap, Output')
plt.xlabel('Neuron')
plt.ylabel('Neuron')
plt.show()
# %%


almost_zero = w2.abs() < 1e-3
plt.figure(figsize=(10, 8))
sns.heatmap(almost_zero, cmap='viridis')
plt.title('abs(weight) < 1e-3')
plt.xlabel('Output Neuron')
plt.ylabel('Hidden Neuron')
plt.show()
# %%


'T1000_D100_L1_tu0.1_bs256_lr0.03'
tu = 0.1
bs = 256
lr = 0.03
hp = HyperParameters(T=1000, D=100, H=None, L=1, 
                                tu=tu, bs=bs, lr=lr,
                                ntype = 'mlp', gtype = 'and')

new_trainer = Trainer(hp, group=group)
new_trainer.train(steps=500, log_interval=10)
save(new_trainer)
new_trainer.train(steps=5000, log_interval=100)
save(new_trainer)
new_trainer.train(steps=15000, log_interval=100)
save(new_trainer)
# %%
