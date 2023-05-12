def func(a, b=2, c=3): ...
#   ^^^^>

def keywords():
  func(6, 93, b=62)  # Noncompliant {{Remove duplicate values for parameter "b" in "func" call.}}
#         ^^  ^^^^<
  func(6, 93, 21, c=62)  # Noncompliant
  func(6, 93, 21, d=62)  # S930 scope

  def keyword_only(a, *, b): ...
  def positional_only(a, /, b): ...
  keyword_only(1, b=2, a=2)  # Noncompliant
  positional_only(1, 2, b=2)  # Noncompliant
  positional_only(1, 2, a=2)  # S930 scope
  positional_only(1, a=2)  # S930 scope

def dict_literals():
  params = {'c': 31}
  func(6, 93, 31, **params)  # Noncompliant
  func(6, 93, c=62, **params)  # Noncompliant
  func(6, 93, c=62, **{'c': 31})  # Noncompliant
  func(6, 93, **{'c': 31, 'c': 32})  # S5780 scope: the resulting dictionary only has 1 element per key

  c = "not c"
  dict_key_not_literal = {c : 1}
  func(1, 2, 3, **dict_key_not_literal) # OK

  params_altered = {'c': 3}
  del params_altered['c']
  func(6, 93, 31, **params_altered)  # OK

  if condition:
    multiple_assignments = {'e': 31}
  else:
    multiple_assignments = {'c': 31}
  func(6, 93, 31, **multiple_assignments)  # Only single assignments are supported

  nested_dict = {**params, 'unknown': 34}
  func(6, 93, 41, **nested_dict) # FN (nested dict not accounted for)

  another_nested_dict = {'b': 42, **params}
  func(**another_nested_dict, a=41, b=43) # Noncompliant

  func(6, 93, 31, **unknown_dict) # OK

def dict_calls():
  params_dict = dict(c=31)
  func(6, 93, c=62, **params_dict)  # Noncompliant
  func(6, 93, c=62, **dict(c=31))  # Noncompliant

  nested_dict_1 = dict(**params_dict, u=42)
  nested_dict_2 = dict(params_dict, u=42)
  func(6, 93, c=62, **nested_dict_1)  # OK
  func(6, 93, c=62, **nested_dict_2)  # OK

  another_nested_dict = dict(**params_dict, b=2)
  func(6, 93, c=62, **another_nested_dict)  # Noncompliant

  unknown_dict = unknown_call(c=31)
  func(6, 93, c=62, **unknown_dict)  # OK
  func(6, 93, c=62, **func(1,2,3))  # OK
  func(6, 93, c=62, **unknown_call(c=31))  # OK
  func(6, 93, c=62, **(a, b))  # for coverage
  tuple = (a, b)
  func(6, 93, c=62, **tuple)  # for coverage

def with_args():
  def func_with_args(a, *args, b=None): ...
  func_with_args(a, b, c, b = 42) # OK

  def func(a, b, c): ...
  my_args = 42, 43
  func(*my_args, 44) # OK
  my_args2 = ()
  func(*my_args2, 42, 43, c=44) # OK

def tuples_no_fp():
  def rectangle((top, left), (width, height)): ...
  rectangle((0, 0), (width, height))
  dimensions = (3, 4)
  rectangle((0, 0), dimensions)

def function_from_typeshed():
  import emoji
  emoji.emojize(":rocket:", True, use_aliases=True) # FN {{Remove duplicate values for parameter "use_aliases" in "emojize" call.}}


def self_ignored():
  class MyClass:
    def method_positional_only(self, /, **kwargs): ...
    def method_regular(self): ...
    obj = MyClass2()
    obj.method_positional_only(self=1) # OK
    obj.method_regular(self=1) # FN

  class MyClass2:
    def method1(self, a): ...
    def method2(self, a):
      # To avoid FPs, no issue on duplicated self
      self.method1(self, a)  # S930 scope
      self.method1(a, self=self) # FN
      self.method1(self, a=a) # Noncompliant
      "{self}".format(self=self)  # OK


def exceptions():
  from zope.interface import Interface
  class IProperties(Interface):
    def setProperty(name, value, source, runtime=False): ...

  props = IProperties(self)
  props.setProperty(propname, value, source, runtime=runtime) # OK

  @some_decorator()
  def decorated(a): ...
  decorated(42, a = 43) # OK

  unknown_func()
  class MyClass:
    def method(self, b): ...
  a = MyClass()
  a.method(b)
  MyClass.method(a, b)
  MyClass.method(a, b=b)

class StaticCallInsideClass:
  def my_method(a, b): ...
  my_method(1,b=2) # OK

def not_static_call():
  class MyClass:
    def foo(self, x): ...
  a = MyClass()
  f = a.foo
  f(42, x=42) # FN
