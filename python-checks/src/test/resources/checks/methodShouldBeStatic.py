class A:

    def fun1(self):
        print(self)

    @classmethod
    def fun2(cls):
        self = 1
        print(self)

    @staticmethod
    def fun3(arg):
        print(arg)

    @staticmethod
    def fun4():
        print()

    def fun5(self, x): # Noncompliant {{Make this method static.}}
#       ^^^^
        print(x)

    def fun6(self, x):
        print(x)
        if True:
            print(self)

    def fun7(self):
        self.x = 5

    def fun8(self): # Noncompliant
        if True:
            print(1)
            return 1

    @another_decorator
    @staticmethod
    def fun9(self):
        print()

    def fun10(slf, x): # Noncompliant
        print(x)

    def fun11(slf, x):
        print(slf.field)

    def fun13(slf, x):
        print(x)
        def fun12():
            print(slf.field)

    @another_decorator
    def fun14(self): # Noncompliant
        print(1)

    def fun15(self):
        '''this method doesnt have implementation and shouldnt raise issue'''

    def fun16(self):
        pass

    def fun17(self):
        '''docstring'''
        pass

    def fun18(self): pass

    def fun19(self):
        if True:
            print(self)
        print(1)
        print(2)

    def fun20(): #syntax error
        print(1)
        
    # don't raise issue, as the method may raise a NotImplementedError   
    def fun21(slf):
        raise NotImplementedError('Temporary stuff')
    def fun22(slf):
        print(1)
        raise NotImplementedError
    def fun23(slf):
        """ Short explanation """
        if True:
            raise NotImplementedError
    def fun29(slf): raise NotImplementedError

    # raise issue, as the NotImplementedError is returned, not thrown
    def fun28(slf): # Noncompliant
        return NotImplementedError
        
    # raise issue, as the error is not a NotImplementedError
    def fun31(slf): raise ValueError('Some error') # Noncompliant
    def fun32(slf): raise                          # Noncompliant
    

    # built-in functions

    def __init__(slf):
        print(1)

    def __init__(slf):
        print(slf)

    def __this_is_not_a_built_in_method__(slf): # this method is not a built-in, although for simplicity our implementation considers it as such
        print(1)

    def _init__(slf, x): # Noncompliant
        print(x)

    def in__it__(slf, x): # Noncompliant
        print(x)

    def foo_bar(self):
        print(self)


class B(A):
    def foo_bar(self):
        print(1)

class IssueExpected(object):
    def foo_bar(self): # Noncompliant
        print(1)

class NoIssueExpected1(object, IssueExpected):
    def foo_bar(self):
        print(1)

class NoIssueExpected2(IssueExpected, A):
    def foo_bar(self):
        print(1)

def fun():
    print(1)

def __abs__(x):
    print(1)
