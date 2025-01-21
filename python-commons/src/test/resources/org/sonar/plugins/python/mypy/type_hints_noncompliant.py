# Example from the mypy documentation

def greet_all(names: list[str]) -> None:
    for name in names:
        print('Hello ' + name)

names = ["Alice", "Bob", "Charlie"]
ages = [10, 20, 30]

greet_all(names)
greet_all(ages)

def no_type_hints(x):
    return [x]

from unknown import unknown

if __name__ == "__name__":
    print(no_type_hints(0))
    greet_all(unknown())

from typing import List

def ignore_comment() -> List[str]: # type: ignore
    return []
