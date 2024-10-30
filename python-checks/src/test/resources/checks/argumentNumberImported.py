def fn(p): pass

class A():
    def meth(self, p1): pass
    def foo(p1): pass


reassigned_unbound_meth = A.meth
reassigned_bound_meth = a().meth
