def foo():
    1l # Noncompliant {{Replace suffix in long integers from lower case "l" to upper case "L".}}
#   ^^
    1L
