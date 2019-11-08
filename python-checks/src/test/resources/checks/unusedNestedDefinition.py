def compliant():
  def using():  # Ok
    nested_function()
    NestedClass()

  def nested_function():  # Ok
    print("nested_function")

  class NestedClass:  # Ok
    def __init__(self):
      print("NestedClass")

  using()


def noncompliant():
  def using():  # Noncompliant {{Remove this unused function declaration.}}
#     ^^^^^
    nested_function()

  class NestedClass():  # Noncompliant {{Remove this unused class declaration.}}
#       ^^^^^^^^^^^
      def __init__(self):
          print("NestedClass")

  def nested_function():  # Ok
    print("nested_function")


def callbacks():
  @d.callback
  def using():  # Ok
    pass

  @d.callback
  class NestedClass:  # Ok
      def __init__(self):
          print("NestedClass")

def func():
  def otherfunc():
    if cond:
      def nested(): #Noncompliant
        pass
  otherfunc()
  pass

class clazz:
  def func():
    class inner:
      def some_method(): # Ok
        print("hello")

    some = inner
    return some

class clazz:
  class nested: # Ok
    pass
