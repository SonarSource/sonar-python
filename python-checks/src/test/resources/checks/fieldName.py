
class MyClass:
    myField = 4 # Noncompliant {{Rename this field "myField" to match the regular expression ^[_a-z][a-z0-9_]+$.}}
#   ^^^^^^^
    myField2: int = 4 # Noncompliant {{Rename this field "myField2" to match the regular expression ^[_a-z][a-z0-9_]+$.}}
#   ^^^^^^^^
    my_field = 4

    def __init__(self):
        localVar = 0
        self.myField = 0
        self.my_field1 = 0
        self.myField1 = 0 # Noncompliant
#            ^^^^^^^^

    def fun(self):
        self.myField.field = 1
        self.newField = 0 # Noncompliant
    fun.skip = True


class MyClass1:
    implements(ICredentialsChecker)
    # Noncompliant@+3
    # Noncompliant@+2
    # Noncompliant@+1
    myField1 = myField2 = myField3 = None
    myField1, newField = 1, 2 # Noncompliant

class MyClass2:
    # Noncompliant@+2
    # Noncompliant@+1
    (Field1, (Field2, f2)) = (1, 2)

    def fun(self):
        (self.Field3, x) = (1, 2) # Noncompliant
        # Noncompliant@+2
        # Noncompliant@+1
        self.Field4 = self.Field5 = 6

class MyClass3(object):
    myField = 4 # Noncompliant

class MyClass4(MyClass3):
    myField = 4

class MyClass5:
    class MyClass6:
        myField1 = 1            # Noncompliant

class MyClass7:
    class MyClass8:
        def foo(self):
            self.myField2 = 2   # Noncompliant

class MyClass7(object, MyClass3):
    myField = 5
