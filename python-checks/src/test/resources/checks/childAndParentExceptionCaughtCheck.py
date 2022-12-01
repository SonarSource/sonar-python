def child_with_parent():
  try:
      raise NotImplementedError()
  except (NotImplementedError, RuntimeError):  # Noncompliant {{Remove this redundant Exception class; it derives from another which is already caught.}}
  #       ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^< {{Parent class.}}
      print("Foo")

  try:
      raise NotImplementedError()
  except* (NotImplementedError, RuntimeError):  # Noncompliant {{Remove this redundant Exception class; it derives from another which is already caught.}}
  #        ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^< {{Parent class.}}
      print("Foo")

def parent_with_child():
    try:
        raise NotImplementedError()
    except (RuntimeError, NotImplementedError):  # Noncompliant {{Remove this redundant Exception class; it derives from another which is already caught.}}
        #   ^^^^^^^^^^^^> ^^^^^^^^^^^^^^^^^^^
        print("Foo")

    try:
        raise NotImplementedError()
    except* (RuntimeError, NotImplementedError):  # Noncompliant
        print("Foo")


def duplicate_exception_caught():
  try:
      raise NotImplementedError()
  except (RuntimeError, RuntimeError):  # Noncompliant {{Remove this duplicate Exception class.}}
#         ^^^^^^^^^^^^  ^^^^^^^^^^^^< {{Duplicate.}}
      print("Foo")

  try:
      raise NotImplementedError()
  except* (RuntimeError, RuntimeError):  # Noncompliant
      print("Foo")

def multiple_parents():
  try:
      raise NotImplementedError()
  except (UnicodeDecodeError, UnicodeError, ValueError):  # Noncompliant 2
      print("Foo")

  try:
      raise NotImplementedError()
  except* (UnicodeDecodeError, UnicodeError, ValueError):  # Noncompliant 2
      print("Foo")

def duplicate_and_parent_with_child():
  try:
      raise NotImplementedError()
  except (RuntimeError, NotImplementedError, RuntimeError):  # Noncompliant 2
      print("Foo")

  try:
      raise NotImplementedError()
  except* (RuntimeError, NotImplementedError, RuntimeError):  # Noncompliant 2
      print("Foo")

def python2_supports_nested_tuples():
    try:
        ...
    except (ValueError, (RuntimeError, NotImplementedError)): # Noncompliant
        ...

    try:
        ...
    except* (ValueError, (RuntimeError, NotImplementedError)): # Noncompliant
        ...

def compliant():
  try:
    raise NotImplementedError()
  except RuntimeError:
    print("Foo")

  try:
    raise NotImplementedError()
  except* RuntimeError:
    print("Foo")

def unknown():
  def method():
    pass

  try:
    raise ValueError()
  except ((A | B), C):
    pass
  except (method, Exception):
    pass
  except:
    pass

  try:
    raise ValueError()
  except* ((A | B), C):
    pass
  except* (method, Exception):
    pass
