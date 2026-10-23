################################################
# Case to cover: Detect when custom types do not
# have the right methods.
################################################

def custom_noncompliant():
  class Empty:
    pass

  class LeftAdd:
    def __add__(self, other):
      return 42

  class RightAdd:
    def __radd__(self, other):
      return 21

  class AddAssign:
    def __iadd__(self, other):
      return 42

  Empty() + 1  # Noncompliant
  LeftAdd() + 1  # Ok
  1 + LeftAdd()  # Noncompliant
  Empty() + LeftAdd()  # Noncompliant
  LeftAdd() + Empty()  # Ok

  RightAdd() + 1  # Noncompliant
  1 + RightAdd()  # Ok
  Empty() + RightAdd()  # Ok
  RightAdd() + Empty()  # Noncompliant


  empty = Empty()
  add_assign = AddAssign()
  empty += 1
  add_assign += 1

  -Empty()  # Noncompliant


def custom_compliant():
  class A:
    def __add__(self, other):
      return 42

    def __neg__(self):
      return -1

  a = A()
  myvar = A() + 1
  myvar = -a


#################################################
# Case to cover: Detect whem builtin types do not
# have the right methods.
#################################################

def builtin_noncompliant():
  1 + "1"  # Noncompliant
  "1" + 1  # Noncompliant
  1 + [1]  # Noncompliant
  1 + {1}  # Noncompliant
  1 + (1,)  # Noncompliant
  1 + {'a': 1}  # Noncompliant
  [1] + (1,)  # Noncompliant
  "foo " + "bar".encode('base64') # OK, FP in Python2
  "bar".encode('base64') + "foo"  # OK, FP in Python2


  -'1'  # Noncompliant

  from queue import Queue
  q = Queue()
  q.maxsize + 'other' # Noncompliant


def builtin_compliant():
  1 + int("1")
  myvar = -int('1')
  myvar += 2

  not myvar # coverage


def type_annotations():
  mode: "OpenBinaryMode" | "OpenTextMode"

#################################################
# Mocks could be monkey patched to possess any special methods
# No issues should be raised on them
#################################################
def mocks():
  from unittest.mock import Mock, MagicMock 
  mock = Mock()
  myvar = Mock() + 1
  myvar = -mock
  mock = MagicMock()
  mock += 1


  class ExtendedMock(MagicMock):
      ...

  def custom_mock():
    ExtendedMock() + 1
    -ExtendedMock()


def type_symbols():
    import ctypes
    class SomeCtypeClass(ctypes.Structure):
        ...
    x = SomeCtypeClass * 42  # OK
    x = 42 * SomeCtypeClass  # OK
    y = ctypes.c_int32 * 2  # OK
    class RandomClass:
        ...
    z = RandomClass * 42 # FN


def type_symbols_try_except():
    import ctypes
    some_variable = None
    some_variable = ctypes.c_int32
    try:
        x = some_variable * 42  # OK
    except:
        x = some_variable * 42  # OK

def http_status_enum():
   from http import HTTPStatus
   HTTPStatus.OK + 1  # OK


from typing import final
from typing_extensions import final as other_final
from importedDecorators import ImportedBase

mark = final

@mark
class Aliased:
    pass

Aliased() + Aliased()  # Noncompliant


def outer():
    local_mark = final
    @local_mark
    class Nested:
        pass
    Nested() + Nested()  # Noncompliant


@(final if condition else other_final)
class KnownUnion:
    pass

KnownUnion() + KnownUnion()  # Noncompliant


def custom(cls):
    cls.__add__ = lambda self, other: 42
    return cls


@(final if condition else custom)
class MixedUnion:
    pass

MixedUnion() + MixedUnion()


@final
class Child(ImportedBase):
    pass

Child() + Child()


@unknown_factory()
class Factory:
    pass

Factory() + Factory()


def decorated_union(condition):
    value = Aliased() if condition else MixedUnion()
    value + value


def plain_function():
    pass

plain_function + 1  # FN: the existing V1 operand inference treats function values as Any.


def nested_invalid_operation():
    MixedUnion() + ("text" + 1)  # Noncompliant


class UnknownBaseChild(unknown_base):
    pass

UnknownBaseChild() + UnknownBaseChild()


class Plain:
    pass


def binary_operations_without_v2_information():
    # V1 knows this is object, but V2 has no type for __getstate__'s result.
    # Keep reporting these operations when refreshed stubs decorate object with typing.disjoint_base.
    state = Plain.__getstate__(Plain())
    state + 1  # Noncompliant
    1 + state  # Noncompliant
    state * state  # Noncompliant

    # Missing V2 information on one side must not bypass the other side's decorators.
    state + MixedUnion()
    MixedUnion() + state
    state + Child()
    Child() + state
    state + UnknownBaseChild()
    UnknownBaseChild() + state


class Directory:
    def __dir__(self):
        return object.__dir__(self) + ["extra"]


def directory_operations(value):
    object.__dir__(value) + ["extra"]
    ["extra"] + object.__dir__(value)
    object.__dir__(value) - 1  # Noncompliant
    object.__dir__(value) + "extra"  # Noncompliant


def directory_alias_operations(value):
    builtin_dir = object.__dir__
    builtin_dir(value) + ["extra"]
    (object.__dir__(value)) + ["extra"]
    -object.__dir__(value)  # Noncompliant
    object.__dir__(value) * 2


class CustomDirectory:
    def __dir__(self) -> tuple:
        return ("custom",)


def custom_directory_operations():
    CustomDirectory().__dir__() + ("extra",)
