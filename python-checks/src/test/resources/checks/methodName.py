class MyClass:
    def correct_method_name():
        pass

    def Incorrect_Method_Name():  # Noncompliant {{Rename method "Incorrect_Method_Name" to match the regular expression ^[a-z_][a-z0-9_]*$.}}
#       ^^^^^^^^^^^^^^^^^^^^^
        pass

    def long_method_name_is_still_correct():
        pass

class MyTestCase(unittest.TestCase):
    def setUp(self): # ok, potentially overridden method, the name can't be changed
        self.message = 'hello'

def This_Is_A_Function():
    pass

class AnotherClass():
    def correct_method(self):
        pass
    def SomeMethod(self): # Noncompliant
#       ^^^^^^^^^^
        pass

class AnotherClass(SomeParent, AnotherParent):
    def A_Method(self): # Potentially overriden method
        pass

class AnotherClass(object):
    def A_Method(self): # Noncompliant [[inherits object]]
        pass
class A():
    if 1:
        def Badly_Named(self): # Noncompliant
            pass
class B(SuperClass):
    if 1:
        def Badly_Named(self): # compliant, might be overriding
            pass


class DatabaseModel:
  @property
  def db(self):  # OK
    raiseNotImplementedError()

class Coordinate:
  @property
  def x(self):  # OK
    return 42

  @property
  def y(self):  # OK
    return 42

  def _v(self):
    ...
