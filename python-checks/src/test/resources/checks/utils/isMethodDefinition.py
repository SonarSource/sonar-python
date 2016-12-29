
globalvar = 5 # Noncompliant {{not_method_definition}}

def a(): # Noncompliant {{not_method_definition}}
    pass

class A: # Noncompliant {{not_method_definition}}
    field = 1

    def a(self): # Noncompliant {{method_definition}}
        pass

    class BB: # Noncompliant {{not_method_definition}}
        field1 = 1
        field2 = 1

        def b(self): # Noncompliant {{method_definition}}
            pass

def b(): # Noncompliant {{not_method_definition}}
    pass
