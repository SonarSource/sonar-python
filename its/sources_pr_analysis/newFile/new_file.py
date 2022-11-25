from submodule import SubmoduleException

try:
    raise SubmoduleException()
except (SubmoduleException, NotImplementedError): # Noncompliant
    print("Foo")
