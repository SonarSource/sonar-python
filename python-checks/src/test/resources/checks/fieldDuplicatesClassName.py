class MyClass:
  myClass = 3      # Noncompliant {{Rename field "myClass"}}
# ^^^^^^^
  def __int__(self, myclass):
    self.myclass: type = myclass # Noncompliant 
#     ^^^^^^^@-4<

  def fun(self):
    self.myClass += 1  # Noncompliant
    if True:
      self.MYCLASS = 10 # Noncompliant
    self.field = 5

  class NestedClass:
#       ^^^^^^^^^^^>
    def fun(self):
      self.nestedClass = 5 # Noncompliant {{Rename field "nestedClass"}} 
#          ^^^^^^^^^^^

class MyClass1(MyClass):
  myClass1 = 1

class MyClass:
  class B:
     myClass = 1
     def foo(self):
         self.myClass = 1

class MyClass:
  if True:
    MYCLASS = 1 # Noncompliant
    def foo(self):
      self.myCLASS = 1 # Noncompliant
      myClass = 1      # compliant, local var

class MyClass2:
  myClass2: int = 3      # Noncompliant {{Rename field "myClass2"}}

class MyClass3:
  def myClass3(self): pass # OK
