def basic_calls():
    round(1.3) # Noncompliant {{The return value of "round" must be used.}}
   #^^^^^
    x = round(1.3)
    x = y = round(1.3)
    round(1.3) + round(1.2)
    round(1.2), round(1.3) # Noncompliant 2
    print(x) # Ok
    no_such_function(1)

def string_calls():
    "hello".capitalize() # Noncompliant {{The return value of "str.capitalize" must be used.}}
    s0 = "hello".capitalize()
    s1 = "hello"
    s1.capitalize() # Noncompliant
    s2 = s1.capitalize()
    s2[0] # Noncompliant
    'll' in s2 # Noncompliant

from collections import defaultdict

class MyDict(dict):
    pass

class MyCollection:
    def __contains__(self, item):
        pass

def collection():
    dict({'a': 1, 'b': 2}) # Noncompliant
    d1 = dict({'a': 1, 'b': 2})
    d1.copy() # Noncompliant

    d1['a'] # Noncompliant {{The return value of "__getitem__" must be used.}}
   #^^^^^^^
    'a' in d1 # Noncompliant {{The return value of "__contains__" must be used.}}
   #^^^^^^^^^
    x = d1['a']
    x = 'a' in d1

    d2 = defaultdict()
    d2['a'] # Ok - defaultdict.__geitem__ creates the key if missing
    d3 = MyDict()
    d3['a'] # FN
    foo = MyCollection
    'a' in foo # Ok

def calling_instance_methods_via_string():
    # Calling instance methods on the class has no side effect either
    str.islower('this is passed as self')  # Noncompliant

def exceptions_in_try_blocks():
    try:
        round(3.14) # Ok
    except ValueError as e:
        round(3.14)  # Noncompliant

    try:
        round(3.14) # Ok
        if x:
            round(3.14) # Ok
        return
    except ValueError as e:
        round(3.14)  # Noncompliant

def edge_case():
    round = 1
    round(1.3)


s = "hello"
s.replace('o', 'x') # Noncompliant
print(s)


def type_calls(param):
    type(param) # Noncompliant
    X = type(param)
    X() # OK, class instantiation could have a side effect


def pytorch_pure_operations():
    import torch
    tensor = torch.tensor([-1.0, 2.0, -3.0])
    tensor.abs()  # Noncompliant {{The return value of "torch.Tensor.abs" must be used.}}
#   ^^^^^^^^^^
    print(tensor)  # Still contains negative values

    # Basic arithmetic operations with in-place equivalents
    tensor.add(1.0)  # Noncompliant {{The return value of "torch.Tensor.add" must be used.}}
    tensor.sub(0.5)  # Noncompliant {{The return value of "torch.Tensor.sub" must be used.}}
    tensor.mul(2.0)  # Noncompliant {{The return value of "torch.Tensor.mul" must be used.}}
    tensor.div(2.0)  # Noncompliant {{The return value of "torch.Tensor.div" must be used.}}
    tensor.pow(2)    # Noncompliant {{The return value of "torch.Tensor.pow" must be used.}}

    # Mathematical functions with in-place equivalents
    tensor.sqrt()    # Noncompliant {{The return value of "torch.Tensor.sqrt" must be used.}}
    tensor.neg()     # Noncompliant {{The return value of "torch.Tensor.neg" must be used.}}
    tensor.exp()     # Noncompliant {{The return value of "torch.Tensor.exp" must be used.}}
    tensor.log()     # Noncompliant {{The return value of "torch.Tensor.log" must be used.}}

    # Trigonometric functions with in-place equivalents
    tensor.sin()     # Noncompliant {{The return value of "torch.Tensor.sin" must be used.}}
    tensor.cos()     # Noncompliant {{The return value of "torch.Tensor.cos" must be used.}}
    tensor.tan()     # Noncompliant {{The return value of "torch.Tensor.tan" must be used.}}
    tensor.sinh()    # Noncompliant {{The return value of "torch.Tensor.sinh" must be used.}}
    tensor.cosh()    # Noncompliant {{The return value of "torch.Tensor.cosh" must be used.}}
    tensor.tanh()    # Noncompliant {{The return value of "torch.Tensor.tanh" must be used.}}

    # Rounding and clamping operations with in-place equivalents
    tensor.floor()   # Noncompliant {{The return value of "torch.Tensor.floor" must be used.}}
    tensor.ceil()    # Noncompliant {{The return value of "torch.Tensor.ceil" must be used.}}
    tensor.round()   # Noncompliant {{The return value of "torch.Tensor.round" must be used.}}
    tensor.trunc()   # Noncompliant {{The return value of "torch.Tensor.trunc" must be used.}}
    tensor.frac()    # Noncompliant {{The return value of "torch.Tensor.frac" must be used.}}
    tensor.clamp(0, 1)  # Noncompliant {{The return value of "torch.Tensor.clamp" must be used.}}

    # Activation functions with in-place equivalents
    tensor.sigmoid() # Noncompliant {{The return value of "torch.Tensor.sigmoid" must be used.}}
    tensor.relu()    # Noncompliant {{The return value of "torch.Tensor.relu" must be used.}}
    tensor.leaky_relu(0.1)  # Noncompliant {{The return value of "torch.Tensor.leaky_relu" must be used.}}

    # Softmax operations with in-place equivalents (though less common)
    tensor.softmax(0)     # Noncompliant {{The return value of "torch.Tensor.softmax" must be used.}}
    tensor.log_softmax(0) # Noncompliant {{The return value of "torch.Tensor.log_softmax" must be used.}}

    mask = torch.tensor([True, False, True])
    tensor.masked_fill(mask, 0.0)  # Noncompliant {{The return value of "torch.Tensor.masked_fill" must be used.}}

    indices = torch.tensor([0, 1, 2])
    tensor.index_fill(0, indices, 1.0)  # Noncompliant {{The return value of "torch.Tensor.index_fill" must be used.}}

    tensor.copy()    # Noncompliant {{The return value of "torch.Tensor.copy" must be used.}}
    tensor.clone()   # Noncompliant {{The return value of "torch.Tensor.clone" must be used.}}
    tensor.detach()  # Noncompliant {{The return value of "torch.Tensor.detach" must be used.}}

    # Correct usage - assigning results
    result = tensor.abs()
    tensor_copy = tensor.clone()
    detached = tensor.detach()

    # In-place operations are OK (they modify the original tensor)
    tensor.abs_()
    tensor.add_(1.0)
    tensor.mul_(2.0)


def pytorch_edge_cases():
    import torch

    zeros = torch.zeros(3, 4)
    zeros.abs()  # Noncompliant

    ones = torch.ones(2, 3)
    ones.neg()   # Noncompliant

    tensor = torch.tensor([1.0, 2.0, 3.0])

    tensor.size()     # OK - returns size, no in-place equivalent
    tensor.dim()      # OK - returns dimension count
    tensor.numel()    # OK - returns number of elements

    tensor.abs().sum()
