import torch.nn
import torchvision.models as models

def noncompliant():
    model = models.vgg16()
    model.load_state_dict(torch.load('model_weights.pth')) # Noncompliant {{Set the module in training or evaluation mode.}}
   #^^^^^^^^^^^^^^^^^^^^^
    ...

class CustomModule(torch.nn.Module):
    pass

def noncompliant():
    model = CustomModule()
    model.load_state_dict(torch.load('model_weights.pth')) # Noncompliant
    ...

def noncompliant():
    model = models.vgg16()
    model.train()
    model.load_state_dict(torch.load('model_weights.pth')) # Noncompliant

def noncompliant():
    model = models.vgg16()
    model.load_state_dict(torch.load('model_weights.pth'))
    model.train()

    model.load_state_dict(torch.load('model_weights.pth')) # Noncompliant

    other_model = model

def compliant(model):
    model.load_state_dict(torch.load('model_weights.pth'))



def compliant():
    model3 = models.vgg16()
    model3.load_state_dict(torch.load('model_weights.pth')) # Ok if model is passed as argument to a function do not raise at all train or eval could be called in such functions
    foo(model3)





