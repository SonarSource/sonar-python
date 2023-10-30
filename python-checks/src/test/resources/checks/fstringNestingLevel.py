
def non_compliant(hello, name):
    my_string = f"{f"{f"{hello}"},"} {name}!" # Noncompliant {{Do not nest f-strings too deeply.}}
    #           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 

    greeting = f"{f"{hello}"},"
    my_string = f"hello:{greeting}, {name}, {f"end: { f"done" }"}!" # Noncompliant
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 

    my_string = f"hello { f"format specifier: {greeting :{F"1"}.{2}}"}" # Noncompliant
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def compliant(hello, name):
    greeting = f"{f"{hello}"},"
    my_string = f"{greeting} {name}!" # Compliant

    my_string = f"{greeting} {name : { f"1" }.{2}}!" 
