class EmptyClass:
    ...

class ClassWithEq:
    def __eq__(self, other):
        return self

class ClassWithHash:
    def __hash__(self):
        return 42

class ClassWithEqInherithingHash(ClassWithHash):
    def __eq__(self, other):
        return self


class HashIsNone:
    __hash__ = None


def set_members():
    {EmptyClass()}  # OK - object has by default a __hash__() method
    {ClassWithEq()} # FN
    {ClassWithEqInherithingHash()} # FN
    my_list = [1, 2, 3]
    {my_list} # Noncompliant {{Make sure this expression is hashable.}}
#    ^^^^^^^
    {1, 2, 3}
    {ClassWithHash()}
    {some_func()}
    {None}
    {HashIsNone()} # FN

def dictionary_keys():
    {EmptyClass(): 42} # OK
    my_list = [1, 2, 3]
    {my_list: 42} # Noncompliant
    {1: "one", 2: "two"}
    {ClassWithHash(): 42}
    {None: 42}


def some_func():
    return "foo"
