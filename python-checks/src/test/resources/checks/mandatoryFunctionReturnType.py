def foo(): # Noncompliant {{Add a return type hint to this function declaration.}}
   #^^^
    return "bar"

class Baz:
    def __init__(): # Noncompliant {{Annotate the return type of this constructor with `None`.}}
       #^^^^^^^^
        pass

    def foo(): # Noncompliant {{Add a return type hint to this function declaration.}}
       #^^^
        return "foo"

def __init__(): # Noncompliant {{Add a return type hint to this function declaration.}}
   #^^^^^^^^
    print("foo")


def nested() -> str:
    def child(): # Noncompliant {{Add a return type hint to this function declaration.}}
       #^^^^^
        return "foo"
    return "foo"

def success() -> str:
    return "bar"

def successUnion() -> str|int:
    return "bar"

class InitSuccess:
    def __init__() -> None:
        pass
