from my_model import model

def dataLoader(param_dict, param_list):
    from torch.utils.data import DataLoader
    noncompliant = DataLoader() # Noncompliant
    noncompliant = DataLoader(model.parameters()) # Noncompliant {{Add the missing hyperparameter batch_size for this PyTorch optimizer.}}
                  #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    optimizer = DataLoader(model.parameters(), 3)
    optimizer = DataLoader(model.parameters(), batch_size=3)
    optimizer = DataLoader(model.parameters(), **param_dict)
    optimizer = DataLoader(model.parameters(), *param_list)

def adadelta(param_dict, param_list, rho, eps):
    from torch.optim import Adadelta
    noncompliant = Adadelta() # Noncompliant {{Add the missing hyperparameters lr and weight_decay for this PyTorch optimizer.}}
    noncompliant = Adadelta(some_extra_variable=3) # Noncompliant {{Add the missing hyperparameters lr and weight_decay for this PyTorch optimizer.}}
    noncompliant = Adadelta(model.parameters(), lr = 0.001) # Noncompliant
    noncompliant = Adadelta(model.parameters(), weight_decay = 0.32) # Noncompliant

    optimizer = Adadelta(model.parameters(), 0.001, rho, eps, 0.23)
    optimizer = Adadelta(model.parameters(), lr = 0.001, weight_decay=0.23)
    optimizer = Adadelta(model.parameters(), lr = 0.001, weight_decay=0.23, some_extra_variable=3)

def adagrad(param_dict, param_list, lr_decay):
    from torch.optim import Adagrad
    noncompliant = Adagrad() # Noncompliant
    noncompliant = Adagrad(model.parameters(), lr = 0.001) # Noncompliant
    noncompliant = Adagrad(model.parameters(), weight_decay = 0.001) # Noncompliant

    optimizer = Adagrad(model.parameters(), 0.001, lr_decay, 0.23)
    optimizer = Adagrad(model.parameters(), lr = 0.001, weight_decay=0.23)

def adam(param_dict, param_list, betas, eps):
    from torch.optim import Adam
    noncompliant = Adam() # Noncompliant
    noncompliant = Adam(model.parameters(), lr = 0.001) # Noncompliant
    noncompliant = Adam(model.parameters(), weigth_decay = 0.001) # Noncompliant

    optimizer = Adam(model.parameters(), 0.001, betas, eps, 0.23)
    optimizer = Adam(model.parameters(), lr = 0.001, weight_decay=0.23)
    optimizer = Adam(model.parameters(), **param_dict, *param_list)
    optimizer = Adam(model.parameters(), **param_dict, *param_list)

def adamW(param_dict, param_list, betas, eps):
    from torch.optim import AdamW
    noncompliant = AdamW() # Noncompliant
    noncompliant = AdamW(model.parameters(), lr = 0.001) # Noncompliant
    noncompliant = AdamW(model.parameters(), weigth_decay = 0.001) # Noncompliant

    optimizer = AdamW(model.parameters(), 0.001, betas, eps, 0.23)
    optimizer = AdamW(model.parameters(), lr = 0.001, weight_decay=0.23)

def sparse_adam(param_dict, param_list):
    from torch.optim import SparseAdam
    noncompliant = SparseAdam() # Noncompliant


    optimizer = SparseAdam(model.parameters(), 0.001)
    optimizer = SparseAdam(model.parameters(), lr = 0.001)


def adamax(param_dict, param_list, betas, eps):
    from torch.optim import Adamax
    noncompliant = Adamax() # Noncompliant
    noncompliant = Adamax(model.parameters(), lr = 0.001) # Noncompliant
    noncompliant = Adamax(model.parameters(), weigth_decay = 0.001) # Noncompliant

    optimizer = Adamax(model.parameters(), 0.001, betas, eps, 0.23)
    optimizer = Adamax(model.parameters(), lr = 0.001, weight_decay=0.23)

def asgd(param_dict, param_list, lambda_, alpha, t0):
    from torch.optim import ASGD
    noncompliant = ASGD() # Noncompliant
    noncompliant = ASGD(model.parameters(), lr = 0.001) # Noncompliant
    noncompliant = ASGD(model.parameters(), weight_decay = 0.001) # Noncompliant

    optimizer = ASGD(model.parameters(), 0.001, lambda_, alpha, t0, 0.23)
    optimizer = ASGD(model.parameters(), lr = 0.001, weight_decay=0.23)

def lbfgs(param_dict, param_list):
    from torch.optim import LBFGS
    noncompliant = LBFGS() # Noncompliant

    optimizer = LBFGS(model.parameters(), 0.001)
    optimizer = LBFGS(model.parameters(), lr = 0.001)

def nadam(param_dict, param_list, betas, eps):
    from torch.optim import NAdam
    noncompliant = NAdam() # Noncompliant {{Add the missing hyperparameters lr, weight_decay and momentum_decay for this PyTorch optimizer.}}
    noncompliant = NAdam(model.parameters(), lr = 0.001) # Noncompliant
    noncompliant = NAdam(model.parameters(), weight_decay = 0.001) # Noncompliant
    noncompliant = NAdam(model.parameters(), momentum_decay = 0.001) # Noncompliant

    optimizer = NAdam(model.parameters(), 0.001, betas, eps, 0.23, 0.25)
    optimizer = NAdam(model.parameters(), lr = 0.001, weight_decay=0.23, momentum_decay=0.25)

def radam(param_dict, param_list, betas, eps):
    from torch.optim import RAdam
    noncompliant = RAdam() # Noncompliant
    noncompliant = RAdam(model.parameters(), lr = 0.001) # Noncompliant
    noncompliant = RAdam(model.parameters(), weight_decay = 0.001) # Noncompliant

    optimizer = RAdam(model.parameters(), 0.001, betas, eps, 0.23)
    optimizer = RAdam(model.parameters(), lr = 0.001, weight_decay=0.23)

def rms_prop(param_dict, param_list, alpha, eps):
    from torch.optim import RMSprop
    noncompliant = RMSprop() # Noncompliant
    noncompliant = RMSprop(model.parameters(), lr = 0.001) # Noncompliant
    noncompliant = RMSprop(model.parameters(), weight_decay = 0.001) # Noncompliant
    noncompliant = RMSprop(model.parameters(), momentum = 0.001) # Noncompliant

    optimizer = RMSprop(model.parameters(), 0.001, alpha, eps, 0.23, 0.323)
    optimizer = RMSprop(model.parameters(), lr = 0.001, weight_decay=0.23, momentum=0.323)

def rprop(param_dict, param_list):
    from torch.optim import Rprop
    noncompliant = Rprop() # Noncompliant
    noncompliant = Rprop(model.parameters()) # Noncompliant

    optimizer = Rprop(model.parameters(), 0.001)
    optimizer = Rprop(model.parameters(), lr = 0.001)

def sgd(param_dict, param_list, dampening):
    from torch.optim import SGD
    noncompliant = SGD() # Noncompliant
    noncompliant = SGD(model.parameters(), lr = 0.001) # Noncompliant
    noncompliant = SGD(model.parameters(), weight_decay = 0.001) # Noncompliant
    noncompliant = SGD(model.parameters(), momentum = 0.001) # Noncompliant

    optimizer = SGD(model.parameters(), 0.001, 0.323, dampening, 0.23)
    optimizer = SGD(model.parameters(), lr = 0.001, momentum=0.323, weight_decay=0.23)
