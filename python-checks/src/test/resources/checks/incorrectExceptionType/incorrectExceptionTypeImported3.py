class SomeException(BaseException):
  pass

class SomeNotException(object):
  pass


class SomeChildException(SomeException):
  pass

class SomeChildNotException(SomeNotException):
  pass

class Enclosing:
  class Nested():
    pass

class Animal:
    pass
