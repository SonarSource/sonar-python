def iterable():
  from collections.abc import Iterable
  def foo() -> Iterable[str]:
    yield ""

def iterator():
  import collections.abc
  def some_method(param) -> collections.abc.Iterator[SomeClass]:
    for entry in param:
      yield entry

def generator():
  import collections.abc
  def foo() -> collections.abc.Generator[str]:
    yield ""

def conditional_import():
  import sys
  if sys.version_info < (3, 9):
    from typing import Iterator
  else:
    from collections.abc import Iterator

  async def my_iterator() -> Iterator[str]:
    yield "hello"
