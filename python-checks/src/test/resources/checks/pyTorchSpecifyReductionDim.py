import torch

def argmin(input, dict_args):
    torch.argmin(input) # Noncompliant {{Provide a value for the dim argument.}}
   #^^^^^^^^^^^^
    torch.argmin(**dict_args) # Noncompliant

    torch.argmin(input, 2)
    torch.argmin(input, dim=2)


def aminmax(input):
    torch.aminmax(input) # Noncompliant
    torch.aminmax(input, 2) #Noncompliant

    torch.aminmax(input, dim=2)

def nanmean(input):
    torch.nanmean(input) # Noncompliant

    torch.nanmean(input, 2)
    torch.nanmean(input, dim=2)

def mode(input):
    torch.mode(input) # Noncompliant

    torch.mode(input, 2)
    torch.mode(input, dim=2)

def norm(input, p):
    torch.norm(input) # Noncompliant
    torch.norm(input, 2) #Noncompliant

    torch.norm(input, p, 2)
    torch.norm(input, dim=2)

def quantile(input, q):
    torch.quantile(input) # Noncompliant
    torch.quantile(input, 2) #Noncompliant

    torch.quantile(input, q, 2)
    torch.quantile(input, dim=2)

def nanquantile(input, q):
    torch.nanquantile(input) # Noncompliant
    torch.nanquantile(input, 2) #Noncompliant

    torch.nanquantile(input, q, 2)
    torch.nanquantile(input, dim=2)

def std(input):
    torch.std(input) # Noncompliant

    torch.std(input, 2)
    torch.std(input, dim=2)

def std_mean(input):
    torch.std_mean(input) # Noncompliant

    torch.std_mean(input, 2)
    torch.std_mean(input, dim=2)

def unique(input, sorted, return_inverse, return_counts):
    torch.unique(input) # Noncompliant
    torch.unique(input, sorted, return_inverse, 2) #Noncompliant

    torch.unique(input, sorted, return_inverse, return_counts, 2)
    torch.unique(input, dim=2)

def unique_consecutive(input, return_inverse, return_counts):
    torch.unique_consecutive(input) # Noncompliant
    torch.unique_consecutive(input, return_inverse, return_counts) #Noncompliant

    torch.unique_consecutive(input, return_inverse, return_counts, 2)
    torch.unique_consecutive(input, dim=2)

def var(input):
    torch.var(input) # Noncompliant

    torch.var(input, 2)
    torch.var(input, dim=2)

def var_mean(input):
    torch.var_mean(input) # Noncompliant

    torch.var_mean(input, 2)
    torch.var_mean(input, dim=2)

def count_nonzero(input):
    torch.count_nonzero(input) # Noncompliant

    torch.count_nonzero(input, 2)
    torch.count_nonzero(input, dim=2)
