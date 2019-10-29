def func1(): # Noncompliant {{Add a way to break out of this function's recursion.}}
#   ^^^^^
    func1()
#   ^^^^^<

def func2():
    func1()

def func3(x):
    x, func3 = lambda y: y + 1
    return func3(x)

def func4(x):
    def func4(x):
        return 2
    return func4(x)

def func5(x): # Noncompliant
#   ^^^^^
    try:
        return func5(x)
#              ^^^^^<
    except:
        return 0

def func6(x):
    try:
        return func5(x)
    except:
        return func6(x + 1)

def func7(x):
    z =  [func7(y) for y in x]
    return len(z)

def func8(x):
    from .other import func8
    return func8(x)

def func9(x):
    from .other import f1 as func1, f2 as func9
    func1()
    return func9(x)

def func10(x): # Noncompliant
    from .other import func10 as f1
    f1()
    return func10(x)

def fun11():
    for fun11 in [lambda: 1, lambda: 2]:
        # fun11 match the for variable and not the function name
        print(fun11())

def fun12():
    # fun12 match the for variable and not the function name
    return [fun12() for fun12 in [lambda: 1, lambda: 2]]

def fun13(x): # Noncompliant
    if x:
        print(1)
    else:
        print(1)
    fun13(x)

def fun14(x): # false-negative
    if x:
        fun14(x)
    else:
        fun14(x)

class A:

    def func1(self):
        func1()

    def func2(self): # Noncompliant {{Add a way to break out of this method's recursion.}}
#       ^^^^^
        self.func1()
        self.func2()
#       ^^^^^^^^^^<

    @staticmethod
    def func3(): # Noncompliant {{Add a way to break out of this method's recursion.}}
#       ^^^^^
        A.func4()
        A.func3()
#       ^^^^^^^<

    @classmethod
    def func4(cls): # Noncompliant {{Add a way to break out of this method's recursion.}}
#       ^^^^^
        A.func4()
#       ^^^^^^^<

    def func5(self):
        x = lambda self: self.func5()
        return 1

    def func6(self):
        class B:
            def func5(self):
                self.func6()

            def func6(self):
                pass

        return 2

    def func7(other_self): # false-negative
        other_self.func1()
        other_self.func2()

    def func8(self):
        self.func8 = lambda: 2
        return self.func8()
