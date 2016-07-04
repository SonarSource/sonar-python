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
        
        
    # From here on, we test the "raise NotImplementedError" case 

    # don't raise issue, as the method consists only of a NotImplementedError   
    def fun21a(slf):
        raise NotImplementedError

    # don't raise issue, as the method consists only of a NotImplementedError   
    def fun21b(slf):
        raise NotImplementedError('Temporary stuff')

    # raise issue, as NotImplementedError is not the only content of the method
    def fun22(slf): # Noncompliant
        print(1)
        raise NotImplementedError

    # raise issue, as NotImplementedError is not the only content of the method
    def fun23(slf): # Noncompliant
        if True:
            raise NotImplementedError

    # raise issue, as the error is not a NotImplementedError 
    def fun24(slf): # Noncompliant
        raise ValueError('Some error')
        
    # don't raise issue, as the method has no argument
    def fun25():
        raise NotImplementedError
                
    # don't raise issue, as we don't handle a doc string as a statement
    def fun26(slf):
        """ Short explanation """
        raise NotImplementedError
                
    # raise issue, as the second literal is not a doc string
    def fun27(slf): # Noncompliant
        """ Long explanation
            bla bla
        """
        """ Another literal """
        raise NotImplementedError
        
    # raise issue, as the error is returned, not thrown
    def fun28(slf): # Noncompliant
        return NotImplementedError
        
    # don't raise issue, as the method consists only of a NotImplementedError   
    def fun29(slf): raise NotImplementedError
    
    # raise issue, as NotImplementedError is not the only content of the method
    def fun30(slf): print(1) ; raise NotImplementedError # Noncompliant
                
    # raise issue, as the error is not a NotImplementedError
    def fun31(slf): raise ValueError('Some error') # Noncompliant
    
    # don't raise issue, as the method consists only of a NotImplementedError   
    def fun25(slf, other):
        # some comment
        raise NotImplementedError

    # from here on, we test built-in functions

    def __init__(slf):
        print(1)

    def __init__(slf):
        print(slf)

    def __this_is_not_a_built_in_method__(slf):
        print(1)

    def _init__(slf, x): # Noncompliant
        print(x)

    def in__it__(slf, x): # Noncompliant
        print(x)

    def foo_bar(self):
        print(self)


class B(A):

    # raises an issue. This is questionable, see SONARPY-166
    def foo_bar(self): # Noncompliant
        print(1)


def fun():
    print(1)

def __abs__(x):
    print(1)
