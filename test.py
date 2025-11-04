# Import for test.py
import torch
import matplotlib.pyplot as plt
from setup import list_saves, load, save, Trainer, HyperParameters# 
# 
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
trainer.train(steps=500, log_interval=10)
trainer.train(steps=5000, log_interval=100)

# %%
trainer.train(steps=5000, log_interval=100)