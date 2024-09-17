import torch
from torch import load
import safetensors

model = torch.load('model.pth') # Noncompliant {{Replace this call with a safe alternative.}}
       #^^^^^^^^^^

some_path = ...
model = load(some_path) # Noncompliant

torch.load(model2, 'model.pth', weights_only=False) #Noncompliant

model2 = ...
safetensors.torch.load_model(model2, 'model.pth')

torch.load(model2, 'model.pth', weights_only=True)

def unknown_weights_only_value(some_value, some_func):
    torch.load(model2, 'model.pth', weights_only=some_value)
    torch.load(model2, 'model.pth', weights_only=some_func())

def conditional_weights_only(cond):
    weights_only = True
    if cond:
        weights_only = False

    torch.load(model2, 'model.pth', weights_only=weights_only)

def only_one_definition():
    weights_only = True
    torch.load(model2, 'model.pth', weights_only=weights_only)

    weights_only = False
    torch.load(model2, 'model.pth', weights_only=weights_only) #Noncompliant

# test if no issue is raised if there is no symbol for the callee
something[42]()