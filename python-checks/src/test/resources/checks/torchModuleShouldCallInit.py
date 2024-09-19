import torch.nn as nn

class NonCompliantModule(nn.Module):
     #^^^^^^^^^^^^^^^^^^> {{Inheritance happens here}}
    def __init__(self): #Noncompliant {{Add a call to super().__init__().}}
       #^^^^^^^^
        ...


class NonCompliantModule(OtherModule, nn.Module):
    def __init__(self): #Noncompliant
        ...

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

class CompliantModule(nn.Module):
    def __init__(self, super):
        super().__init__()

class CompliantModule(nn.Module):
    def __init__(self):
        (lambda x: x)()
        (lambda x: x)().test()
        self.call_super()

    def call_super(self):
        super().__init__()


class CompliantModule(nn.Module):
    def __init__(self):
        pass

    def call_super(self):
        super().__init__()

class CompliantModule2(nn.Module):
    def __init__(self): #FN
        do_something()

    class Nested(Other):
        def __init__(self):
            super().__init__()

class UnrelatedCompliantModule:
    def __init__(self):
        ...

def __init__(test):
    pass

def some_other_func():
    pass

class CompliantModule(nn.Module):
    if cond:
        class some_func: pass
    else:
        def some_func(): pass
