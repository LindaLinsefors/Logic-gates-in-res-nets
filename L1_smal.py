# %%
# Setup for L1_smal.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
import setup
import importlib
import seaborn as sns
import numpy as np

importlib.reload(setup)
from setup import (ResNet, LogicGates, HyperParameters,Trainer, _show_matrix, 
                   save, list_saves, load,
                   show_resnet_L1, 
                   several_ouptupt_featueres_contur_plots,
                   all_ouptupt_featueres_contur_plots,
                   get_active_relus,
                   show_matrix,
                   )


torch.set_default_device("cuda") # Set default device

wandb.login() # Login to wandb

group = 'Res L1'

# all files have the same gates
all_files = list_saves(group)
all_files = [f for f in all_files if 'T80_' in f]

# %%
'''
T = 80
D = 50
gates = load(group, all_files[0]).gates

for tu in [0, 0.01, 0.1]:
    for H in [50]:
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
'''
# %%
# %%
trainer = load(group, 'T80_D50_H50_L1_nai3_tu0.1_bs2048_wd0.01_lr0.001_s35500_3xzn6oy9')
# %%
trainer.train(steps=15000, log_interval=100)
save(trainer)
trainer.train(steps=15000, log_interval=100)
save(trainer)
trainer.train(steps=15000, log_interval=100)
save(trainer)



# %%

files = [f for f in all_files if '_H50_' in f and '_s35500_' in f]
trainers = [load(group, f) for f in files]
trainers = sorted(trainers, key=lambda tr: tr.hp.tu)


# %%
for tr in trainers:
    print(f'Target uncertainty {tr.hp.tu}')
    show_resnet_L1(tr)

# %%

for tr in trainers:
    several_ouptupt_featueres_contur_plots(tr, title=f'tu = {tr.hp.tu}')

# %%
tr = trainers[-1]

out_features = torch.arange(0, tr.hp.T)

list_same = []
list_always_on = []
list_agree = []
relus_used_by_output = []

for of in out_features:
    input = torch.zeros(4, tr.hp.T)
    input[(0,1), tr.gates.first_inputs[of]] = 1
    input[(0,2), tr.gates.second_inputs[of]] = 1
    target = tr.gates.forward(input.T).T[:, of]
    with torch.no_grad():
        output = tr.model(input)[:, of]
    active_relus = get_active_relus(tr, input)[0]

    same = (active_relus[0][None,:] == active_relus[1:]).all(dim=0)
    always_on = active_relus.all(dim=0)
    agree = (same == always_on).all()

    if agree == False:
        print('Disagreement between always on and same!')
    else:
        relus_used_by_output.append(~same)

    list_same.append(same)
    list_always_on.append(always_on)
    list_agree.append(agree)

relus_used_by_output = torch.stack(relus_used_by_output)


relus_per_output = relus_used_by_output.sum(dim=1)
outputs_per_relu = relus_used_by_output.sum(dim=0)

plt.subplot(1,2,1)
plt.hist(relus_per_output.cpu(), bins=30)
plt.xlabel('Number of relus used by output feature')
plt.ylabel('Number of output features')
plt.title('Relus per output feature')

plt.subplot(1,2,2)
plt.hist(outputs_per_relu.cpu(), bins=30)
plt.xlabel('Number of output features using ReLU')
plt.ylabel('Number of ReLUs')
plt.title('Output features per ReLU')

plt.show()

# %%
files = [f for f in all_files if '_3xzn6oy9' in f]
trainers = [load(group, f) for f in files]
trainers = sorted(trainers, key=lambda tr: tr.current_step)

for tr in trainers:
    print(f'Step {tr.current_step}')

for tr in (trainers[3], trainers[-1]):
    show_resnet_L1(tr)
# %%

_show_matrix(relus_used_by_output.cpu())
plt.grid(True)
plt.xlabel('ReLU index')
plt.show()

relu_to_output = combine_linear(tr.model.hidden_layers[0].layer[2],
                                tr.model.output_layer,)
_show_matrix(relu_to_output.weight.data.cpu())
plt.grid(True)
plt.xlabel('ReLU')
plt.show()

# %%
_show_matrix(relu_to_output.weight.data.cpu())
plt.plot([1,2,3,4,5], [1,2,3,4,5], marker='x', linestyle='None')
plt.grid(True)
plt.xlabel('ReLU index')
plt.show()

# %%
relus = []
outputs = []
for relu in range(tr.hp.H):
    for output in range(tr.hp.T):
        if relus_used_by_output[output, relu]:
            relus.append(relu + 0.5)
            outputs.append(output + 0.5)

_show_matrix(relus_used_by_output.cpu())
plt.plot(relus, outputs, marker='x', 
         linestyle='None', color='black', markersize=2)
plt.grid(True)
plt.xlabel('ReLU index')
plt.show()

# %%
plt.figure(figsize=(12,10))
_show_matrix(relu_to_output.weight.data.T.cpu())
plt.plot(outputs, relus, marker='o', linestyle='None', 
         markerfacecolor='black', markeredgecolor='white', markersize=5)

plt.grid(True)
plt.ylabel('ReLU')
plt.xlabel('Output')
plt.show()
# %%
