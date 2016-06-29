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

    def fun5(self, x):
        print(x)

    def fun6(self, x):
        print(x)
        if True:
            print(self)

    def fun7(self):
        self.x = 5

    def fun8(self):
        if True:
            print(1)
            return 1

    @another_decorator
    @staticmethod
    def fun9(self):
        print()

    def fun10(slf, x):
        print(x)

    def fun11(slf, x):
        print(slf.field)

    def fun13(slf, x):
        print(x)
        def fun12():
            print(slf.field)

    @another_decorator
    def fun14(self):
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
    def fun22(slf):
        print(1)
        raise NotImplementedError

    # raise issue, as NotImplementedError is not the only content of the method
    def fun23(slf):
        if True:
            raise NotImplementedError

    # raise issue, as the error is not a NotImplementedError 
    def fun24(slf):
        raise ValueError('Some error')
        
    # don't raise issue, as the method has no argument
    def fun25():
        raise NotImplementedError
                
    # don't raise issue, as we don't handle a doc string as a statement
    def fun26(slf):
        """ Short explanation """
        raise NotImplementedError
                
    # raise issue, as the second literal is not a doc string
    def fun27(slf):
        """ Long explanation
            bla bla
        """
        """ Another literal """
        raise NotImplementedError
        
    # raise issue, as the error is returned, not thrown
    def fun28(slf):
        return NotImplementedError
        
    # don't raise issue, as the method consists only of a NotImplementedError   
    def fun29(slf): raise NotImplementedError
    
    # raise issue, as NotImplementedError is not the only content of the method
    def fun30(slf): print(1) ; raise NotImplementedError
                
    # raise issue, as the error is not a NotImplementedError
    def fun31(slf): raise ValueError('Some error')
    
def fun():
    print(1)