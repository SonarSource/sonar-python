class Context:
    def __exit__(self, exc_type, exc_value, traceback):
        raise # Noncompliant {{Remove this "raise" statement and return "False" instead.}}
       #^^^^^
        raise exc_value # Noncompliant
       #^^^^^^^^^^^^^^^
        raise exc_type

class DoNotMatchOnExit:
    def foo(self):
        raise

def not_called_exit():
    raise

class TupleExit:
    def __exit__((self, exc_type, exc_value, traceback)):
        raise exc_value

class AnotherContext:
    def __exit__(self, exc_type, *, exc_value, traceback):
        raise exc_value

class PackedContext:
    def __exit__(self, *args):
        raise args[2] # Noncompliant
        raise args[3] # Noncompliant
        raise self[2]
        raise foo()[2]

class PackedContextWithoutName:
    def __exit__(self, *):
        raise # Noncompliant

class PackedLikeParameterWithoutStarToken:
    def __exit__(self, args):
        pass

class NoExitParamsContext:
    def __exit__(self):
        pass

def __exit__():
    pass


class MyContextManager:
    def __enter__(self, stop_exceptions):
        return self
    def __exit__(self, *args):
        try:
            print("42")
        except:
            print("exception")
            raise  # No issue when raising another exception. The __exit__ method can fail and raise an exception
        raise MemoryError("No more memory")  # This is ok too.