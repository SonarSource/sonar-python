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

def fun():
    print(1)