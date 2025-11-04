# %%
# Import for tu.py
import torch
import matplotlib.pyplot as plt
from setup import list_saves, load, save, Trainer, HyperParameters


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











# %%
# Reload trainers at step 11000

files = list_saves(group)
files_s11000 = [f for f in files if '_s11000_' in f]
for f in files_s11000:
    print(f)

trainers_tu = {}
trainers_tu[0] = [load(group, f) for f in files_s11000 if '_tu0_' in f]
trainers_tu[0.01] = [load(group, f) for f in files_s11000 if '_tu0.01_' in f]
trainers_tu[0.1] = [load(group, f) for f in files_s11000 if '_tu0.1_' in f] 

# %%
# Plot outputs of full network vs input and output layers only

for tu in [0, 0.01, 0.1]:
    for i in range(3):
        net = trainers_tu[tu][i].model

        w_in = net.input_layer.weight.data
        w_out = net.output_layer.weight.data

        b_in = net.input_layer.bias.data
        b_out = net.output_layer.bias.data

        inp = torch.eye(6)
        out_simple = net.output_layer(net.input_layer(inp))
        out = net.forward(inp)

        plt.plot(out_simple.flatten().detach().cpu().numpy(), 
                 out.flatten().detach().cpu().numpy(), 
                 'o',label=f'tu={tu}, run={i}')
    plt.legend()
    plt.xlabel('Output/Input Layer Only')
    plt.ylabel('Full Network')
    plt.title('Output Comparison')
    plt.show()


# %%
# Plot weight and bias distributions

'''
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
'''

for tu in [0, 0.01, 0.1]:
    w_in = [t.model.input_layer.weight.data for t in trainers_tu[tu]]
    w_in = torch.stack(w_in).flatten().cpu().numpy()
    plt.hist(w_in, bins=50, alpha=0.5, label=f'tu={tu}', density=True)
plt.legend()
plt.title('Input Layer Weights Distribution at Step 11000')
plt.xlabel('Weight Value')
plt.ylabel('Density')
plt.show()
for tu in [0, 0.01, 0.1]:
    b_in = [t.model.input_layer.bias.data for t in trainers_tu[tu]]
    b_in = torch.stack(b_in).flatten().cpu().numpy()
    plt.hist(b_in, bins=50, alpha=0.5, label=f'tu={tu}', density=True)
plt.legend()
plt.title('Input Layer Biases Distribution at Step 11000')
plt.xlabel('Bias Value')
plt.ylabel('Density')
plt.show()

for tu in [0, 0.01, 0.1]:
    w_1_in = [t.model.hidden_layers[0].layer[0].weight.data for t in trainers_tu[tu]]
    w_1_in = torch.stack(w_1_in).flatten().cpu().numpy()
    plt.hist(w_1_in, bins=50, alpha=0.5, label=f'tu={tu}', density=True)
plt.legend()
plt.title('First Residual Block Input Weights Distribution at Step 11000')
plt.xlabel('Weight Value')
plt.ylabel('Density')
plt.show()
for tu in [0, 0.01, 0.1]:
    b_1_in = [t.model.hidden_layers[0].layer[0].bias.data for t in trainers_tu[tu]]
    b_1_in = torch.stack(b_1_in).flatten().cpu().numpy()
    plt.hist(b_1_in, bins=50, alpha=0.5, label=f'tu={tu}', density=True)
plt.legend()
plt.title('First Residual Block Input Biases Distribution at Step 11000')
plt.xlabel('Bias Value')
plt.ylabel('Density')
plt.show()


for tu in [0, 0.01, 0.1]:
    w_1_out = [t.model.hidden_layers[0].layer[2].weight.data for t in trainers_tu[tu]]
    w_1_out = torch.stack(w_1_out).flatten().cpu().numpy()
    plt.hist(w_1_out, bins=50, alpha=0.5, label=f'tu={tu}', density=True)
plt.legend()
plt.title('First Residual Block Output Weights Distribution at Step 11000')
plt.xlabel('Weight Value')
plt.ylabel('Density')
plt.show()
for tu in [0, 0.01, 0.1]:
    b_1_out = [t.model.hidden_layers[0].layer[2].bias.data for t in trainers_tu[tu]]
    b_1_out = torch.stack(b_1_out).flatten().cpu().numpy()
    plt.hist(b_1_out, bins=50, alpha=0.5, label=f'tu={tu}', density=True)
plt.legend()
plt.title('First Residual Block Output Biases Distribution at Step 11000')
plt.xlabel('Bias Value')
plt.ylabel('Density')
plt.show()


for tu in [0, 0.01, 0.1]:
    w_2_in = [t.model.hidden_layers[1].layer[0].weight.data for t in trainers_tu[tu]]
    w_2_in = torch.stack(w_2_in).flatten().cpu().numpy()
    plt.hist(w_2_in, bins=50, alpha=0.5, label=f'tu={tu}', density=True)    
plt.legend()
plt.title('Second Residual Block Input Weights Distribution at Step 11000')
plt.xlabel('Weight Value')
plt.ylabel('Density')
plt.show()
for tu in [0, 0.01, 0.1]:
    b_2_in = [t.model.hidden_layers[1].layer[0].bias.data for t in trainers_tu[tu]]
    b_2_in = torch.stack(b_2_in).flatten().cpu().numpy()
    plt.hist(b_2_in, bins=50, alpha=0.5, label=f'tu={tu}', density=True)
plt.legend()
plt.title('Second Residual Block Input Biases Distribution at Step 11000')
plt.xlabel('Bias Value')
plt.ylabel('Density')
plt.show()

for tu in [0, 0.01, 0.1]:
    w_2_out = [t.model.hidden_layers[1].layer[2].weight.data for t in trainers_tu[tu]]
    w_2_out = torch.stack(w_2_out).flatten().cpu().numpy()
    plt.hist(w_2_out, bins=50, alpha=0.5, label=f'tu={tu}', density=True)
plt.legend()
plt.title('Second Residual Block Output Weights Distribution at Step 11000')
plt.xlabel('Weight Value')
plt.ylabel('Density')
plt.show()
for tu in [0, 0.01, 0.1]:
    b_2_out = [t.model.hidden_layers[1].layer[2].bias.data for t in trainers_tu[tu]]
    b_2_out = torch.stack(b_2_out).flatten().cpu().numpy()
    plt.hist(b_2_out, bins=50, alpha=0.5, label=f'tu={tu}', density=True)
plt.legend()
plt.title('Second Residual Block Output Biases Distribution at Step 11000')
plt.xlabel('Bias Value')
plt.ylabel('Density')
plt.show()

for tu in [0, 0.01, 0.1]:
    w_out = [t.model.output_layer.weight.data for t in trainers_tu[tu]]
    w_out = torch.stack(w_out).flatten().cpu().numpy()
    plt.hist(w_out, bins=50, alpha=0.5, label=f'tu={tu}', density=True)
plt.legend()
plt.title('Output Layer Weights Distribution at Step 11000')
plt.xlabel('Weight Value')
plt.ylabel('Density')
plt.show()

# %%
# Plot logits vs targets for different target uncertainty levels


for tu in [0, 0.01, 0.1]:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for i in range(3):
        net = trainers_tu[tu][i].model
        gates = trainers_tu[tu][i].gates

        inputs = gates.generate_input_data_cuda(2048)
        with torch.no_grad():
            targets = gates.forward_cuda(inputs).float()

        outputs = net.forward(inputs.T)

        for j in [0,1]:
            ax[j].hist(outputs[targets.T==j].detach().cpu(),
                       bins=5, density=True, alpha=0.5, label=f'run={i}')
        

    ax[0].set_title(f'Logits Distribution for Target=0, tu={tu}')
    ax[1].set_title(f'Logits Distribution for Target=1, tu={tu}')
    for a in ax:
        a.set_xlabel('Logit Value')
        a.set_ylabel('Density')
        a.legend()
    plt.show()




# %%
