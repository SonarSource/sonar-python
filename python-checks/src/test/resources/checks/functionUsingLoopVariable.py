def simple_reference():
  for i in range(10):
    print(i)


def simple_reference2():
  return [j for j in range(5)]


def default_value():
  """Passing the variable's value via a default value is ok."""
  mylist = []
  for i in range(5):
    mylist.append(lambda i=i: i)

    def func(i=i):
      return i
    mylist.append(func)


def not_using_outer_variables():
  """Shadowing outer variables with parameters is ok."""
  mylist = []
  for j in range(5):
    print(j)
    mylist.append(lambda j: j)
    def func(j):
      return j
    mylist.append(func)


def return_lambda(param):
  """Exception: returning a lambda/function makes it ok.
  The variable will not change its value as there are no more iterations.
  """
  for j in range(5):
    if param:
      return lambda: j
    elif x:
      def func():
        return j
      return func
    else:
      lamb = lambda: j
      return lamb
  return lambda: 42

def multiple_assignment_yield():
  x = 42
  if cond:
    yield x
  x = bar()
  yield x
  for i in range(10):
    x = foo(bar(lambda: i))
    yield x

def variable_ref_and_func_ref():
  """Referencing a variable updated in the enclosing loop and passing the function via reference is suspicious."""
  mylist = []
  for j in range(5):  # Secondary location on "j", it is the only code updating j.
#     ^> {{Assignment in the loop}}
    mylist.append(lambda: j)  # Noncompliant {{Add a parameter to the parent lambda function and use variable "j" as its default value; The value of "j" might change at the next loop iteration.}}
#                         ^
#                 ^^^^^^@-1< {{Lambda capturing the variable}}
    def func():
      return j  # Noncompliant
    mylist.append(func)


def list_comprehension_lambda_referenced():
  """Referencing a variable updated in the enclosing comprehension and returning a reference to the function is suspicious."""
  # Secondary location on "j" of "for j"
  return [lambda: j for j in range(5)]  # Noncompliant
#         ^^^^^^> ^     ^<


def list_comprehension_lambda_called():
  """Referencing a variable updated in the enclosing comprehension and calling the function in the comprehension is ok.
  This should happen rarely but is a special case of functions called in loops.
  """
  return [(lambda: j)(5) for j in range(5)]  # OK


def all_iterating_variable():
  """All referenced variable should be considered."""
  mylist = []
  for i, j in zip(range(5), range(5)):
    computed = j * 2  # Secondary location on "computed", it is the only code updating computed.
#   ^^^^^^^^> {{Assignment in the loop}}
    mylist.append(lambda: computed)  # Noncompliant
    mylist.append(lambda: j)  # Noncompliant

    def func1():
#       ^^^^^> {{Function capturing the variable}}
      return computed # Noncompliant {{Add a parameter to function "func1" and use variable "computed" as its default value;The value of "computed" might change at the next loop iteration.}}
#            ^^^^^^^^
    mylist.append(func1)

    def func2():
      return j # Noncompliant
    mylist.append(func2)


def generator_expression():
  """Generator expressions are ok when they don't reference outer variables."""
  return (j for j in range(5))

def lambda_in_generator(param):
  """lambdas defined in generator expressions are suspicious enven if it could raise False Positives.
  """
  gen = (lambda: i for i in range(3))  # Noncompliant
  if param:
    return [func() for func in gen]  # This would return [0, 1, 2]
  else:
    funcs = [func for func in gen]
    return [func() for func in funcs]  # This would return [2, 2, 2]


def function_called_in_loop():
  """Exception: Lambda/Functions defined and called in the same loop are ok.
  Having this exception means that we won't detect issues if a different iteration calls the function.
  This False Negative is ok.
  """
  for i in range(10):
    print((lambda param: param * i)(42))
    def func(param):
      return param * i
    print(func(42))


def lambda_variable_called_in_loop():
  """We should make an exception when the lamda is saved in a variable and called in the same loop."""
  for j in range(5):
    lamb = lambda: j  # OK
    lamb()


def lambda_variable_not_called_in_loop():
  """If the lambda is saved in a variable which is not called, then an issue should be raised."""
  mylist = []
  for j in range(5):
    lamb = lambda: j  # Noncompliant
    mylist.append(lamb)
  return mylist


def lambda_variable_called_outside_the_loop():
  for j in range(5):
#     ^>
    if j == 2:
      lamb = lambda: j  # Noncompliant
#                    ^
#            ^^^^^^@-1<
  print(lamb())


def no_binding_usage_in_loop():
  x = 42
  for _ in range(10):
    def func():
      foo(x)

def lambda_assigned_to_tuple():
  for j in range(10):
    (a, b) = (lambda: j, foo)
    return a

def returned_lambda_multiple_assignment():
  for j in range(10):
    var = 42
    var = lambda: j # OK
    return var

def returned_lambda_multiple_assignment_fn():
  for j in range(10):
    if param:
      var = lambda: j # FN
    else:
      var = 42
      return var

class ClassEnclosingScope:
  for j in range(10):
    def func(self):
      return self.j # Noncompliant

#Global scope
for j in range(10):
  def func():
    return j # Noncompliant


def no_issue_when_using_nonlocal():
    for i in range(10):
        var = None
        def foo():
            nonlocal var, i  # OK
            var = 42
    foo()

   for i in range(10):
           var = None
           def foo2():
               global var, i  # OK
               var = 42
       foo()


def no_issue_when_using_default_value():
    """
    The best way to use a loop variable in a function
    s to pass it as a default value.
    """
    for i in range(10):
        def nested1(i=i):  # Ok
            return i

        def nested2(i=[i]):
            return i

        def nested2(i=i.foo):
            return i

        def nested2(i=i + 1):
            return i


def no_issue_for_decorators():
    for i in range(10):
        decorator = lambda func: func
        @decorator  # OK
        def foo():
            pass

        decorator2 = lambda param: lambda func: func
        @decorator2(i)  # OK
        def bar():
            pass

def comprehension_lambda_referenced(value):
    """
    Referencing a variable updated in the enclosing
    comprehension and returning a reference to the function is suspicious.
    """
    if value == "list":
        return [lambda: j for j in range(5)]  # Noncompliant
    elif value == "set":
        return {lambda: j for j in range(5)}  # Noncompliant
    elif value == "dict":
        return {j: lambda: j for j in range(5)}  # Noncompliant
    return 42

