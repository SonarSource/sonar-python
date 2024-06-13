from mypackage.module3 import Mod3Class, Mod3Parent


def hello():
    m3_inst = Mod3Class()
    m3_par_inst = Mod3Parent()
    m3_inst.m3p_meth()
    m3_par_inst.m3p_meth()