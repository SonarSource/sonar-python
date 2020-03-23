class Foo:
    def instance_method(): # Noncompliant {{Add a "self" or class parameter}}
#   ^^^^^^^^^^^^^^^^^^^^^
        pass

    @classmethod
    def class_method(): # Noncompliant {{Add a class parameter}}
#   ^^^^^^^^^^^^^^^^^^
        pass

    def __new__(): # Noncompliant {{Add a class parameter}}
        pass

    def __init_subclass__(): # Noncompliant {{Add a class parameter}}
        pass

    @staticmethod
    def static_method(): # OK, static method
        pass

    def old_style_static_method():
        pass
    old_style_static_method = staticmethod(old_style_static_method)

    def old_style_class_method(): # FN
        pass
    old_style_class_method = classmethod(old_style_class_method)

    def _called_in_cls_body(): # OK, this is called in the class body
        return 1

    x = _called_in_cls_body()

    def referenced_in_cls_body():
        return 1

    options = [referenced_in_cls_body]

    # FP
    def referenced_outside(): # Noncompliant
        return 2

    def no_pos_args(*, kw): # Noncompliant
        pass

    def varargs(*args, kwarg): # OK, unlimited number of positional args
        pass

outside = [Foo.referenced_outside]

import zope.interface as zi
class MyInterface(zi.Interface):
    def do_something(): # OK, it is a zope interface
        pass

class EdgeCase:
    def foo(): # FNs
        pass

EdgeCase = 1

