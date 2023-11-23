import sys
if sys.version_info < (3, 9):
   from typing import Iterator
else:
  from collections.abc import Iterator

async def my_iterator() -> Iterator[str]:
  yield "hello"
