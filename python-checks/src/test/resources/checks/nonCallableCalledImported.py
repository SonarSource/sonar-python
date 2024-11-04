class MyNonCallableClass:
    ...

MyNonCallableClassAlias = MyNonCallableClass

class MyCallableClass:
    def __call__(self):
        ...
