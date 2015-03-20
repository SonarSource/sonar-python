
def fun(input_par1, inputPar2, inputPar3 = 3):
    someName = 1
    another_name = someName
    someName = 3
    inputPar2 = 2



def fun2():
    x = 1
    for i in range(3):
        print(x)
        x = i

def fun3((inputPar2, input_par3), inputPar1=global_var):
    pass

def fun4():
    x = 1
    def fun5():
        x = 2; y = 2
        print(1)

def fun6():
    (a, (b, c)) = (1, (2, 3))
    d, e = 1, 2


def fun7(*ID, **ID2):
    ID += (2,)
    print(ID)
    print(ID2)


def fun8():
    x = MyClass()
    x.b = 2
    x = [1]
    for i in range(1):
        x[i] = 3

def fun9():
    CONSTANT_NAME = "Hello, world"
    for counterName in names:
        pass

def fun10():
    a = b = 1
    name = d.e = 1
