from mod import predicate

class A:
  def make_a(self): ...

class B:
  def make_b(self): ...

if predicate():
  def overloaded_foo(param1: A, param2: int) : ...
#     ^^^^^^^^^^^^^^>
else:
  def overloaded_foo(param1: B, param2: int): ...
#     ^^^^^^^^^^^^^^>

overloaded_foo(42, 42)  # Noncompliant {{Change this argument; Function "overloaded_foo" expects a different type}}
#              ^^


if predicate():
  def overloaded_bar(param1: A, param2: A): ...
else:
  def overloaded_bar(param1: B, param2: B): ...

# we report only on the first noncompliant argument
overloaded_bar(42, 42)  # Noncompliant 2


if predicate():
  def overloaded_baz(param1: A): ...
else:
  def overloaded_baz(param1: B): ...

overloaded_baz(A())  # OK


if predicate():
  def overloaded_fn(param1: A, param2: A): ...
else:
  def overloaded_fn(param1: B, param2: B): ...

overloaded_fn(A(),  # Noncompliant
#             ^^^
                   B())  # Noncompliant
#                  ^^^
