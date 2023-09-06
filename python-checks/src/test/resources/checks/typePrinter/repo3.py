class Myclass1:
    def func(self):
        print("Hello I am class1")
class Myclass2:
    def func(self):
        print("Hello I am class2")


def f3():
    z = 2
    if z > 5:
        x = Myclass1()
    else:
        x = Myclass2()

    x.func()
