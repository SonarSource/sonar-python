def foo():
    1l # Noncompliant {{Replace suffix in long integers from lower case "l" to upper case "L".}}
#   ^^
    1L

qix = [1l] # Noncompliant
bar = -1l # Noncompliant
self.enc.sendEncoded(-1015l) # Noncompliant
plop = 1 ** 2l # Noncompliant
