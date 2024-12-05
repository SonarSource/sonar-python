from typing import Sequence, Union
class Cursor:
    def execute(self, operation: str, parameters: Union[Sequence, None] = None
                ): # should return Cursor
        ...

    def executemany(self, operation: str,
                    seq_of_parameters: Sequence[Union[Sequence, None]]): # should return Cursor
        ...
