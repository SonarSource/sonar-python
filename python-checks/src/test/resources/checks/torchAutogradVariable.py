def torch_import():
    import torch

    x6 = Variable(torch.tensor([15]))

    x = torch.autograd.Variable(  # Noncompliant {{Replace this call with a call to "torch.tensor".}}
    #   ^^^^^^^^^^^^^^^^^^^^^^^
        torch.tensor([1.0]),
        requires_grad=True,
    )
    x2 = torch.autograd.Variable(torch.tensor([1.0]))  # Noncompliant

    x4 = torch.autograd.Variable()  # Noncompliant

    # Compliant solution
    c_x3 = torch.tensor([1.0])


def torch_autograd_import_as():
    from torch.autograd import Variable as V

    x3 = V(torch.tensor([15]))  # Noncompliant
#        ^


def torch_alias_import():
    import torch as t

    x5 = t.autograd.Variable(torch.tensor([15]))  # Noncompliant


def unrelated_import():
    from something.autograd import Variable

    x6 = Variable(torch.tensor([15]))


def multiple_imports():
    from something.autograd import Variable

    x6 = Variable(torch.tensor([15]))
    from torch.autograd import Variable

    x6 = Variable(torch.tensor([15]))  # Noncompliant


def use_before_assignment():
    x6 = Variable(torch.tensor([15]))
    from torch.autograd import Variable

    x6 = Variable(torch.tensor([15]))  # Noncompliant
