from typing import deque, defaultdict, OrderedDict, Counter, ChainMap

import collections

def foo(p: deque[int]): # Noncompliant {{Use the built-in generic type `collections.deque` instead of its typing counterpart.}}
          #^^^^^^^^^^
    pass

def foobar() -> defaultdict[str, int]: # Noncompliant {{Use the built-in generic type `collections.defaultdict` instead of its typing counterpart.}}
               #^^^^^^^^^^^^^^^^^^^^^
    pass

def nested() -> dict[str, list[OrderedDict[str, int]]]: # Noncompliant {{Use the built-in generic type `collections.OrderedDict` instead of its typing counterpart.}}
                              #^^^^^^^^^^^^^^^^^^^^^
    pass

def with_var() -> Counter[int]: # Noncompliant {{Use the built-in generic type `collections.Counter` instead of its typing counterpart.}}
                 #^^^^^^^^^^^^
    my_var: ChainMap[str, int] = None # Noncompliant
           #^^^^^^^^^^^^^^^^^^
    pass

class Success:
    def foo(p: collections.deque[int]):
        pass

    def foobar() -> collections.defaultdict[str, int]:
        pass

    def nested() -> dict[str, list[collections.OrderedDict[str, int]]]:
        pass

    def with_var() -> collections.Counter[int]:
        my_var: collections.ChainMap[str, int] = None
        pass
