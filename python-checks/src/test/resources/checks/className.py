class MyClassWithCompliantName1:
	pass

class MyClass_WithNotCompliantName1: # Noncompliant {{Rename class "MyClass_WithNotCompliantName1" to match the regular expression ^_?([A-Z_][a-zA-Z0-9]*|[a-z_][a-z0-9_]*)$.}}
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	pass

class myClassWithNotCompliantName2: # Noncompliant
	pass


class __FooBar:  # OK
    pass

class my_decorator:  # OK
  def __enter__(self):
    ...
  def __exit__(self, type, value, traceback):
    ...
