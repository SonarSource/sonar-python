import torch
import torchvision.models as models

def noncompliant():
    model = models.vgg16()
    model.load_state_dict(torch.load('model_weights.pth')) # Noncompliant {{Set the module in training or evaluation mode.}}
   #^^^^^^^^^^^^^^^^^^^^^
    ...

def noncompliant(model):
    model.load_state_dict(torch.load('model_weights.pth')) # Noncompliant

def noncompliant():
    model.load_state_dict(torch.load('model_weights.pth')) # Noncompliant

def noncompliant():
    get_model().load_state_dict(torch.load('model_weights.pth')) # Noncompliant

def noncompliant(model):
    weights = torch.load('model_weights.pth')
    weights2 = weights
    model.load_state_dict(weights2) # Noncompliant

def noncompliant():
    model = models.vgg16()
    model.train()
    model.load_state_dict(torch.load('model_weights.pth')) # Noncompliant
    other_model = model

def compliant(model):
    weights = weights
    model.load_state_dict(weights)

def compliant():
    model1 = models.vgg16()
    model1.load_state_dict(torch.load('model_weights.pth'))
    model1.eval()

def compliant():
    model2 = models.vgg16()
    model2.load_state_dict(torch.load('model_weights.pth'))
    other_model = model2
    model2.train()

def compliant():
    model3 = models.vgg16()
    model3.load_state_dict(torch.load('model_weights.pth')) # Ok if model is passed as argument to a function do not raise at all train or eval could be called in such functions
    foo(model3)

def compliant():
    # Ok since no torch.load() result is passed as an argument
    model.load_state_dict(1 + 1)
    model.load_state_dict((lambda x: x)())
