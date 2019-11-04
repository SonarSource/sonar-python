# Noncompliant@+2 {{Add a nested comment explaining why this function is empty, or complete the implementation.}}
bar()
def empty_function():
  pass

def non_empty_function():
  foo()

# Noncompliant@+2
#FP: Comment is associated with previous dedent
def fp():
  pass

foo()
# Not yet implemented - comment associated with def keyword
def func():
  pass

def comment_before_pass():
  # Not yet implemented
  pass

def same_line_as_pass():
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

#FN: some unrelated comment after dedent is attached to the previous function
bar()

class clazz():

  @abstractmethod
  def abstract_method():
    pass

  @abc.abstractmethod
  def abstract_method():
    pass

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
