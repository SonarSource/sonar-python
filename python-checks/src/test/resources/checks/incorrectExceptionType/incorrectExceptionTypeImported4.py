class BaseException():
  pass

class RedefinedBaseExceptionChild(BaseException):
  pass

class ChildOfActualException(Exception):
  pass
