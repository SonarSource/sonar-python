class A:
  __foo = 42 # OK, raised by S4487

  def __unused(self): ... # OK, raised by S1144

  class __unused_cls: ... # Noncompliant {{Remove this unused private '__unused_cls' class.}}
  #     ^^^^^^^^^^^^

  class _unused_cls: ... # Noncompliant

  @unknown
  class _unused_decorated_cls: ...

  class __used_cls: ...

  class __other_used_cls: ...

  def __init__(self):
    print(self.__used_cls)
    print(A.__other_used_cls)
