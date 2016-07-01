class MyClass:
  myClass = 3      # Noncompliant {{Rename field "myClass"}}
# ^^^^^^^
  def __int__(self, myclass):
    self.myclass = myclass # Noncompliant [[secondary=-4]]

  def fun(self):
    self.myClass += 1
    if True:
      self.MYCLASS = 10 # Noncompliant
    self.field = 5

  class NestedClass:
    def fun(self):
      self.nestedClass = 5 # Noncompliant [[secondary=-2]] {{Rename field "nestedClass"}}
#          ^^^^^^^^^^^

class MyClass1(MyClass):
  myClass1 = 1
