
class MyClass:
    myField = 4
    my_field = 4

    def __init__(self):
        localVar = 0
        self.myField = 0
        self.my_field1 = 0
        self.myField1 = 0

    def fun(self):
        self.myField.field = 1
        self.newField = 0
    fun.skip = True


class MyClass1:
    implements(ICredentialsChecker)
    myField1 = myField2 = myField3 = None
    myField1, newField = 1, 2

class MyClass2:
    (Field1, (Field2, f2)) = (1, 2)

    def fun(self):
        (self.Field3, x) = (1, 2)
        self.Field4 = self.Field5 = 6
