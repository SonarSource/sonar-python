from typing import Union

def foo(arg: Union[int, str]):
    if isinstance(arg, int):
        return arg + 1
    else:
        return arg.upper()
