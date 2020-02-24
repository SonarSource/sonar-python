import unknown

class ACustomException(BaseException):
  pass

class SomeClass():
  pass

class SomeDerivedClass(SomeClass):
  pass

class DerivedClassFromUnknown(unknown.Something):
  pass

class Enclosing():
  def __init__(self):
    print("Hello")
  class Nested(BaseException):
    pass
  class Nested2():
    pass

class DerivedFromPython2Exception(StandardError):
  pass
