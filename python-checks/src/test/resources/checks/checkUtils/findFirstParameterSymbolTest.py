class C:
    def hasSymbolFirst01(self):
        ...

    def hasSymbolFirst02(self, sth):
        ...

    def hasNoSymbolFirst01():
        ...

    def hasNoSymbolFirst02((x, y)): # Python 2
        ...

    def hasNoSymbolFirst03(*, x):
        ...

def hasSymbolFirst03(*args):
    ...

def hasSymbolFirst04(**kwargs):
    ...

def hasSymbolFirst05(sth):
    ...
