# Noncompliant@+2 {{Add a nested comment explaining why this function is empty, or complete the implementation.}}
bar()
def empty_function():
  pass

def non_empty_function():
  foo()

#Not yet implemented
def comment_on_previous_dedent():
  pass

foo()
# Not yet implemented
def comment_on_def_keyword():
  pass

def comment_in_body():
  # Not yet implemented
  pass

def comment_same_line_as_pass():
  pass # Not yet implemented

def comment_at_func_def(): # Not yet implemented
  pass

bar()
def comment_after_pass():
  pass
  # Not yet implemented


bar()
def empty_function_fn():
  pass

#FN: some unrelated comment after dedent is linked to the previous function
bar()

foo() # Fixme (FN)

def fn():
  pass

class clazz():

  @abstractmethod
  def abstract_method(self):
    pass

  @abc.abstractmethod
  def abstract_method(self):
    pass

  @abc.abstractproperty
  def abstract_method(self):
    pass

  @abc.abstractclassmethod
  def abstract_method(cls):
    pass

  @abc.abstractstaticmethod
  def abstract_method():
    pass

  foo()
  # Noncompliant@+2 {{Add a nested comment explaining why this method is empty, or complete the implementation.}}
  bar()
  def emtpy_method():
    pass

  def empty_method_with_docstring():
    '''
    Some docstring
    '''

def function_with_multiple_statements():
  foo()
  bar()

class clazz:
  # Not implemented
  def comment_on_indent():
    pass

  if cond:
    #Not implemented
    def  comment_on_indent():
      pass

