from module4 import Mod4Class


class Mod3Parent:
    def m3p_meth(self):
        ...


class Mod3Class(Mod3Parent):
    ...


def foo():
    m4_class = Mod4Class()
    m4_class.m4_meth()
