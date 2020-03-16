# using an assignment expression (:=) as an assignment statement (=) is more explicit
(v := f(p))  # Noncompliant {{Use an assignment statement ("=") instead; ":=" operator is confusing in this context.}}
#^^^^^^^^^
v0 = (v1 := f(p))  # Noncompliant {{Use an assignment statement ("=") instead; ":=" operator is confusing in this context.}}
#     ^^^^^^^^^^

v = f(p)
v0 = v1 = f(p)
x = 42 + (y:=2) # OK, assignment expression within a larger expression.

# using an assignment expression in a function call when keyword arguments are also used.
func(a=(b := f(p)))  # Noncompliant {{Move this assignment out of the argument list; ":=" operator is confusing in this context.}}
func(a := f(p), b=2)  # Noncompliant {{Move this assignment out of the argument list; ":=" operator is confusing in this context.}}
func((a := f(p)) + 42, b=2)  # Noncompliant {{Move this assignment out of the argument list; ":=" operator is confusing in this context.}}
func(a := f(p), 3) # OK
func(a := f(p), (3, 4), *args) # OK
def f(a=(x:=2)+42, b=1): # Noncompliant {{Move this assignment out of the function definition; ":=" operator is confusing in this context.}}
    pass

value = f(p)
func(a=value)
func(value, b=2)

def func(param=(p := 21)):  # Noncompliant
    pass

def func(param=21):
    p = 21

# using an assignment expression in an annotation
def func(param: (p := 21) = 3):  # Noncompliant {{Move this assignment out of the function definition; ":=" operator is confusing in this context.}}
    pass
def f(param: (x:=2)+42, b=1):    # Noncompliant {{Move this assignment out of the function definition; ":=" operator is confusing in this context.}}
    pass
p = 21
def func(param: p = 3):
    pass

# using assignment expression in an f-string. Character ":" is also used as a formatting marker in f-strings.
f'{(x:=10)}'  # Noncompliant
f'{foo(x:=10)}' # Noncompliant
#      ^^^^^

f'{foo(x:=10) + bar(y:=42)}' # Noncompliant
#      ^^^^^        ^^^^^<
f'{x:=10}' # No issue raised but still not recommended. This is not an assignment expression. '=10' is passed to the f-string formatter.

x = 10
f'{x}'
