class clazz:
  def method(self):
#     ^^^^^^>
    foo()
    bar()

  def method2(self): #Noncompliant {{Update this function so that its implementation is not identical to method on line 2.}} 
#     ^^^^^^^
    foo()
    bar()

  def method3(self): #Noncompliant {{Update this function so that its implementation is not identical to method on line 2.}} 
#     ^^^^^^^
#     ^^^^^^@-11<
    foo()
    bar()


class clazz:
  def method(self):
    bar()
    foo()

  def method2(self):
    foo()
    bar()

class clazz:
  def method(self):
#     ^^^^^^>
    if cond:
      foo()
    else:
      bar()

  def method(self): # Noncompliant
#     ^^^^^^
    if cond:
      foo()
    else:
      bar()

class clazz:
  def method(self):
    foo()

  def method2(self): # ok, exception for 1 line methods
    foo()


class clazz:
  def __enter__(self):
    'doc'
    pass

  def __exit__(self): # ok, docstring is not counted as a line of code
    'doc'
    pass

class clazz:
  def __enter__(self):
    'doc'

  def __exit__(self): # ok, docstring is not counted as a line of code
    'doc'

class clazz:
  def __enter__(self):
    '''
    Some multiline
    docstring
    '''
    pass

  def __exit__(self): # ok, docstring is not counted as a line of code
    '''
    Some multiline
    docstring
    '''
    pass

class clazz:
  def method(self):
    class nested:
      def nestedMethod(self):
        foo()
        bar()

  def method(self): #ok, not the same class
    foo()
    bar()

class clazz:
  def method(self):
    def nested():
      foo()
      bar()

  def method(self): #ok, nested is not a method
    foo()
    bar()

class clazz:
  def not_implemented_method(self):
    raise NotImplementedError("Some message")

  def not_implemented_duplicate(self):
    raise NotImplementedError("Some message")

class clazz:
  def not_implemented_method(self):
    'docstringc'
    raise NotImplementedError("Some message")

  def not_implemented_duplicate(self):
    'docstringc'
    raise NotImplementedError("Some message")

class clazz:
  def not_implemented_method(self):
    raise NotImplementedError(
      "Some message"
    )

  def not_implemented_duplicate(self):
    raise NotImplementedError(
      "Some message"
    )

class clazz:
  def not_implemented_method_with_docstring(self):
    '''
    This is a docstring
    '''
    raise NotImplementedError(
      "Some message"
    )

  def not_implemented_duplicate(self):
    '''
    This is a docstring
    '''
    raise NotImplementedError(
      "Some message"
    )

class clazz:
  def multiple_statements_one_line(self):
    'doc'; foo(); bar()

  def multiple_statements_one_line_dup(self): # exception for one line methods
    'doc'; foo(); bar()

class class_with_class_methods():
  @classmethod
  def method_1(cls):
    # ^^^^^^^^>
    print(10)
    print(20)

  @classmethod
  def method_2(cls):  # Noncompliant
    # ^^^^^^^^
    print(10)
    print(20)

  @classmethod
  def foo(cls, first):
    # ^^^>
    print(30)
    print(first)

  @classmethod
  def bar(cls, first):# Noncompliant
    # ^^^
    print(30)
    print(first)

