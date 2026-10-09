from typing import Sequence, Union
class MySQLCursor:
    def __enter__(self) -> "MySQLCursor":
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        ...

    def execute(self, operation: str, parameters: Union[Sequence, None] = None
                ): # should return Cursor
        ...

    def executemany(self, operation: str,
                    seq_of_parameters: Sequence[Union[Sequence, None]]): # should return Cursor
        ...
