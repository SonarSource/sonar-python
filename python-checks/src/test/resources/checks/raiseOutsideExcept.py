def foo():
  raise # Noncompliant {{Remove this "raise" statement or move it inside an "except" block.}}
# ^^^^^
  raise Error()

def __exit__():
  raise

def bar():
  try:
    ...
  except:
    raise
    def fn():
      raise # Noncompliant
  finally:
    raise

def function_called_inside_except():
  def inner():
    raise
  try:
    ...
  except:
    inner()

def function_called_inside_except_group():
  def inner():
    raise
  try:
    ...
  except* OSError:
    inner()

def function_not_called_inside_except():
  def inner():
    raise # Noncompliant
  inner()

class ClassWithMethodCalledFromNestedFunction:
  def inner(self):
    raise

  def call_inner_inside_nested_function(self):
    def callback():
      try:
        ...
      except:
        self.inner()

    callback()

class ClassWithMethodCalledInsideExcept:
  def inner(self):
    raise

  def call_inner_inside_except(self):
    self.inner()
    inner()
    other.inner()
    try:
      ...
    except:
      self.other()
      self.inner()
      get_other().inner()

class ClassWithUnconventionalReceiver:
  def inner(this):
    raise

  def call_inner_inside_except(this):
    try:
      ...
    except:
      this.inner()

class ClassWithClassmethodReceiver:
  def inner(self):
    raise

  @classmethod
  def call_inner_inside_except(cls):
    try:
      ...
    except:
      cls.inner()

class ClassWithClassQualifiedCall:
  def inner(self):
    raise

try:
  ...
except:
  ClassWithClassQualifiedCall.inner()

class ClassWithStaticmethodCaller:
  def inner(self):
    raise # Noncompliant

  @staticmethod
  def call_inner_inside_except(other):
    try:
      ...
    except:
      other.inner()

class ClassWithUnresolvedQualifier:
  def inner(self):
    raise # Noncompliant

try:
  ...
except:
  undefined_obj.inner()

class ClassWithLocalFunction:
  def method(self):
    def inner():
      raise # Noncompliant
    try:
      ...
    except:
      self.inner()

class OuterClass:
  def inner(self):
    raise # Noncompliant

  class NestedClass:
    def call_inner_inside_except(self):
      try:
        ...
      except:
        self.inner()

raise # Noncompliant
