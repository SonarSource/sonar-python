def foo():
    `num` # Noncompliant {{Use "repr" instead.}}
#   ^^^^^
    foo()
    repr("a")
