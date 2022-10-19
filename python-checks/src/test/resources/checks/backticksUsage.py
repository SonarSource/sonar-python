def foo():
    `num` # Noncompliant {{Use "repr" instead.}}
#   ^^^^^
    foo()

def bar():
    print(`1    # Noncompliant
#         ^[el=+2;ec=8]
    + 2`)
    print(`aabb`)   # Noncompliant
#         ^^^^^^
    print(`3 + 4`)  # Noncompliant
#         ^^^^^^^
