s = "Hello world."
s = "Hello \world." # Noncompliant {{Remove this "\", add another "\" to escape it, or make this a raw string.}}
#   ^^^^^^^^^^^^^^^
t = "Nice to \\ meet you"
t = "Nice to \ meet you" # Noncompliant
u = r"Let's have \ lunch"
u = "Let's have \ lunch" # Noncompliant
v = 'Hello world.'
v = 'Hello \world.' # Noncompliant
w = """Hello \n world."""
w = """Hello \w world.""" # Noncompliant
