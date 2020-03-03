class BareRaiseInExit:
    def __exit__(self, exc_type, exc_value, traceback):
        raise # Noncompliant {{Remove this "raise" statement and return "False" instead.}}
       #^^^^^

class ReRaisingExceptionValue:
    def __exit__(self, exc_type, exc_value, traceback):
        raise exc_value # Noncompliant
       #^^^^^^^^^^^^^^^

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

# Edge cases

class RaisingOtherParamThanException:
    def __exit__(self, exc_type, exc_value, traceback):
        raise exc_type

class ClassWithMethodNotCalledExit:
    def foo(self):
        raise

def not_called_exit():
    raise

class ExitWithTupleParameters:
    def __exit__((self, exc_type, exc_value, traceback)):
        raise exc_value

# Invalid __exit__ signatures
class ExitWithMoreParameters:
    def __exit__(self, exc_type, exc_value, traceback, dummy):
        raise exc_value

class ExitWithMoreParameters:
    def __exit__(self, exc_type, exc_value, traceback, dummy):
        raise # Noncompliant

class ExitWithNamedParameters:
    def __exit__(self, exc_type, *, exc_value, traceback):
        raise exc_value

class ExitWithNamedParametersInvalid:
    def __exit__(self, exc_type, *, exc_value):
        raise exc_value

class ExitWithPackedParameters:
    def __exit__(self, *args):
        raise args[2] # Noncompliant

class ExitWithPackedParameters2:
    def __exit__(self, *args):
        raise args[3] # Noncompliant

class ExitWithPackedParameters3:
    def __exit__(self, *args):
        raise self[2]

class ExitWithPackedParameters4:
    def __exit__(self, *args):
        raise foo()[2]

class PackedContextWithoutName:
    def __exit__(self, *):
        raise # Noncompliant

class PackedLikeParameterWithoutStarToken:
    def __exit__(self, args):
        pass

class NoExitParams:
    def __exit__(self):
        pass

def __exit__():
    pass
