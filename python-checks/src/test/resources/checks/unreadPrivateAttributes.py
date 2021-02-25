class A:
  __foo = 42 # Noncompliant {{Remove this unread private attribute '__foo' or refactor the code to use its value.}}
# ^^^^^
  __foo = 43
# ^^^^^< {{Also modified here.}}
  foo = 42

  __bar = 42

  def __unused(self):
      pass

  other_attr = 42

  __slots__ = [] # OK

  def __init__(self):
    self.__attr = 0 # Noncompliant
#        ^^^^^^
    print(A.__bar)
    print(self.other_attr)
