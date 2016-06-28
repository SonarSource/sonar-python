narg=len(sys.argv)
if narg == 1:
    print('@Usage: input_filename nelements nintervals')
    break # Noncompliant {{Remove this "break" statement}}
#   ^^^^^

for i in range(1):
    print(i)
    break

for i in range(1):
    def some_function():
        print(1)
        continue #Noncompliant {{Remove this "continue" statement}}
