def test_direct_interpolation():
    name = "world"
    result = str(t"Hello {name} test")  # Noncompliant {{Template strings should not be used for simple string formatting.}}
    #            ^^^^^^^^^^^^^^^^^^^^

def test_two_t_strings():
    name = "world"
    result = str(t"hello" t"world")  # Noncompliant 2 

def test_deferred_interpolation(): # not yet supported by the check
    name = "world"
    tstring = t"Hello {name}"  
    result = str(tstring)  # FN will be fixed by SONARPY-3471


def test_f_string():
    name = "world"
    result = f"Hello {name}"  # Compliant

def test_t_string_custom_function():
    name = "world"
    from my_module import my_function
    result = my_function(t"Hello {name}")  # Compliant
