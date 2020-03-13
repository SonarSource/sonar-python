def func1(): # Noncompliant {{Add a way to break out of this function's recursion.}}
#   ^^^^^
    func1()
#   ^^^^^<

def func2(a):
    func1()
    a.func1()
    def func1(): # Noncompliant {{Add a way to break out of this function's recursion.}}
    #   ^^^^^
        func1()
    #   ^^^^^<

def func3(x):
    func3 = lambda y: y + 1
    return func3(x)

def func4(x):
    def func4(x):
        return 2
    return func4(x)

def func5(x): # OK
    try:
        return func5(x)
    except:
        return 0

def func6(x):
    try:
        return func5(x)
    except:
        return func6(x + 1)

def func71(x):
    z =  [func71(y) for y in x]
    z =  [1 + func71(y) for y in x]
    return len(z)

def fun72():
    # fun72 match the for variable and not the function name
    return [fun72() for fun72 in [lambda: 1, lambda: 2]]

def fun73(): # Noncompliant
#   ^^^^^
    return [x + 1 for x in fun73()]
#                          ^^^^^<

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

def fun13(x): # Noncompliant
    if x:
        print(1)
    else:
        print(1)
    fun13(x)

def fun14(x): # Noncompliant
#   ^^^^^
    if x:
        fun14(x)
#       ^^^^^<
    else:
        fun14(x)
#       ^^^^^<

def fun15(x):
    if x == 10 or fun15(x + 1):
        fun15(x + 2)
    return x == 10 or fun15(x + 1)

def fun16(x):
    if x == 10 and not fun16(x + 1):
        fun16(x + 2)
    return x == 10 and not fun16(x + 1)

def func17(x): # Noncompliant
#   ^^^^^^
    try:
        print(1)
    except:
        print(2)
    finally:
        func17(x + 1)
#       ^^^^^^<

def func18(x):
    [func18(a) if a == 2 else func18(a + 1) for a in x]
    [func18(a) if func18(a) is not None else func18(a) for a in x]
    [func18(a) for i in x if i%2!=0]
    v = {a : func18(a) for i in x if i%2!=0}
    return (func18(a) for i in x if i%2!=0)

def func19(condition, x):
    return func19(False, x) if condition else 0

def func20(condition, x): # false-negative, ConditionalExpression is not broken down in the cfg
    return func20(False, x) if condition else func20(False, x)

def fun21(x): # Noncompliant
    if fun21(x + 1) or x == 10:
        print(2)

def fun22(r):
    for x in r:
        fun22(r)

def fun23(r): # Noncompliant
    for x in r:
        fun23(r)
    else:
        fun23(r)

def fun24(r):
    for x in r:
        if x:
            break
    else:
        fun24([1])

def fun25(r): # Noncompliant
    for x in r:
        print (x)
    else:
        fun25([1])

class A:

    def func0():
        func0()

    def func1(self):
        func1()
        a = B()
        a.self.func1()

    @unknow
    def func2(self): # Noncompliant {{Add a way to break out of this method's recursion.}}
#       ^^^^^
        self.func1()
        self.func2()
#       ^^^^^^^^^^<

    @staticmethod
    def func3(x): # Noncompliant {{Add a way to break out of this method's recursion.}}
#       ^^^^^
        func3()
        x.func3()
        A.func4()
        A.func3()
#       ^^^^^^^<

    @classmethod
    def func4(cls): # Noncompliant {{Add a way to break out of this method's recursion.}}
#       ^^^^^
        func4()
        cls.func4()
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

    def func7(unexpected_name): # Noncompliant
        unexpected_name.func7()

    def func8(self):
        self.func8 = lambda: 2
        return self.func8()

    @staticmethod
    def func9(x):
        A = lambda: 2
        return A.func9()

    def func10(self):
        self = lambda: 2
        return self.func10()

    def func11(*args):
        args[0].func11()

    def func12((a, b)):
        a.func12()

    def func13(*(a, b)):
        a.func13()

    def func14(self, x):
        self.func14() # false-negative, assignment is after the call
        self = x

    def func15(self, x):
        if x:
            self = x
        self.func15()

    def func16(self, x):
        self.func16() # false-negative, assignment is after the call
        self.func16 = x

    def func17(self, x):
        if x:
            self.func17 = x
        self.func17()

    if True:
        def func18():
            func18()

        def func19(self): # Noncompliant
            self.func19()

        @staticmethod
        def func20():
            func20()

        @staticmethod
        def func21(): # Noncompliant
            A.func21()

# coverage

def func100():
    # invalid cfg
    continue

def func101(x):
    if x:
        func101()
    # invalid cfg
    continue

def func102(a, b, d):
    for i in d:
        if b:
            func102(a, b + i, i)
        else:
            a.end()

def func103(a, b):
    for c in b:
        func103(c, b.child())

def func104(x, a):
    def v1(_):
        func104(x)
    a.add(v1)
    v2 = lambda _: func104(x)
    a.add(v2)
    class C104:
        @staticmethod
        def m(x):
            func104(x)
    a.add(C104())

class C105:
    def func105(self, a):
        def v1(_):
            self.func105()
        a.add(v1)
        v2 = lambda _: self.func105()
        a.add(v2)


def recursion_in_try_block():
    def f():
        try:
            compute()
            f() # OK, compute could raise exception
        except Exception as e:
            print(e)

def async_functions():
    async def f(): # Noncompliant
        await f()

def global_variable(text):
    global global_variable
    global_variable = print
    global_variable(text)

async def asyn(i): # Noncompliant
    print(i)
    return await asyn(i - 1)

async def run():
    var = await asyn(1000000)
    for i in range(1000000):
        var = await var  # OK, await on a call to another function.

async def asyn2(i):
    print(i)
    return asyn2(i - 1) # OK, no await call

async def asyn3(i): # OK, no unconditional await call
  print(i)
  if smth:
    asyn3(i - 1)
  else:
    await asyn3(i - 1)

async def asyn4(i): # Noncompliant
  print(i)
  if smth:
    asyn4(i - 1)
    await asyn4(i - 1)
  else:
    await asyn4(i - 1)

async def asyn5(i): # OK, await is not on "asyn5" call
  await some_call(asyn5(i-1), 42)

async def asyn6(i): # Noncompliant
  await some_call(await asyn6(i-1), 42)

class A:
  def func(*, x, y): ...
