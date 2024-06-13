from subpackage1.module2 import Mod2Class


class Mod4Class:
    def m4_meth(self):
        ...


class Mod4FromMod2(Mod2Class):
    def mod4_m2_meth(self):
        ...
