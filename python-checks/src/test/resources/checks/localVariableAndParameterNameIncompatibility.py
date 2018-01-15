# Noncompliant@+1 {{Rename this parameter "inputPar2" to match the regular expression ^[_a-z][a-z0-9_]+$.}}
def fun(input_par1, inputPar2, inputPar3 = 3): # Noncompliant {{Rename this parameter "inputPar3" to match the regular expression ^[_a-z][a-z0-9_]+$.}}
    someName = 1 # Noncompliant {{Rename this local variable "someName" to match the regular expression ^[_a-z][a-z0-9_]+$.}}
#   ^^^^^^^^
    CamelName: int = 1 # Noncompliant {{Rename this local variable "CamelName" to match the regular expression ^[_a-z][a-z0-9_]+$.}}
#   ^^^^^^^^^
    another_name = someName
    someName = 3
    inputPar2 = 2



def fun2():
    x = 1 # Noncompliant
    for i in range(3):
        print(x)
        x = i
# Noncompliant@+1
def fun3((inputPar2, input_par3), inputPar1=global_var): # Noncompliant
    pass

def fun4():
    x = 1 # Noncompliant
    def fun5():
        # Noncompliant@+2
        # Noncompliant@+1
        x = 2; y = 2
        print(1)

def fun6():
    # Noncompliant@+3
    # Noncompliant@+2
    # Noncompliant@+1
    (a, (b, c)) = (1, (2, 3))

    # Noncompliant@+2
    # Noncompliant@+1
    d, e = 1, 2

# Noncompliant@+2
# Noncompliant@+1
def fun7(*ID, **ID2):
    ID += (2,)
    print(ID)
    print(ID2)


def fun8():
    x = MyClass() # Noncompliant
    x.b = 2
    x = [1]
    for i in range(1):
        x[i] = 3

def fun9():
    CONSTANT_NAME = "Hello, world"
    for counterName in names: # Noncompliant
        pass

def fun10():
    # Noncompliant@+2
    # Noncompliant@+1
    a = b = 1
    name = d.e = 1
