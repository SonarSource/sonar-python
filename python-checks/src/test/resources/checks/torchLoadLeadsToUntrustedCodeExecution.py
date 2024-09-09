import torch
from torch import load
import safetensors

model = torch.load('model.pth') # Noncompliant {{Replace this call with a safe alternative}}
       #^^^^^^^^^^

some_path = ...
model = load(some_path) # Noncompliant

model2 = ...
safetensors.torch.load_model(model2, 'model.pth')

# tests if the rule doesn't crash if the callee symbol is null
something[42]()