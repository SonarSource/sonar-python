def evaluate(command, file, mode):
    eval(command) # Noncompliant
#   ^^^^^^^^^^^^^
    exec(code) # Noncompliant
#   ^^^^^^^^^^
    eval.f() # OK
    myModule.eval() # OK
    myEval(code) # OK
