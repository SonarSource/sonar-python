import torch.nn as nn

class NonCompliantModule(nn.Module):
     #^^^^^^^^^^^^^^^^^^> {{Inheritance happens here}}
    def __init__(self): #Noncompliant {{Add a call to super().__init__()}}
       #^^^^^^^^
        ...

class NonCompliantModule(nn.Module):
    def __init__(self): #Noncompliant
        (lambda x: x)()
        (lambda x: x)().test()
        self.call_super()

    def call_super(self):
        super().__init__()

class NonCompliantModule(OtherModule, nn.Module):
    def __init__(self): #Noncompliant
        self.call_super()

    def call_super(self):
        super().__init__()

class CompliantModule(NonExistantClass):
    def __init__(self, encoder, decoder):
        ...

class CompliantModule(nn.Module):
    def __init__(self):
        super().__init__()
class CompliantModule(nn.Module):
    pass

class CompliantModule(nn.Module):
    def __init__(self, cond):
        if cond:
            super().__init__()

class UnrelatedCompliantModule:
    def __init__(self):
        ...

def __init__(test):
    pass

def some_other_func():
    pass