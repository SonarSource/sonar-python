# using an assignment expression (:=) as an assignment statement (=) is more explicit
(v := f(p))  # Noncompliant
v0 = (v1 := f(p))  # Noncompliant

# using an assignment expression in a function call when keyword arguments are also used.
func(a=(b := f(p)))  # Noncompliant
func(a := f(p), b=2)  # Noncompliant
func(a := f(p), 3) # OK
func(a := f(p), (3, 4), *args) # OK
def func(param=(p := 21)):  # Noncompliant
    pass

# using an assignment expression in an annotation
def func(param: (p := 21) = 3):  # Noncompliant
    pass

# using assignment expression in an f-string. Character ":" is also used as a formatting marker in f-strings.
f'{(x:=10)}'  # Noncompliant
f'{x:=10}' # No issue raised but still not recommended. This is not an assignment expression. '=10' is passed to the f-string formatter.


v = f(p)
v0 = v1 = f(p)

value = f(p)
func(a=value)
func(value, b=2)
def func(param=21):
    p = 21

p = 21
def func(param: p = 3):
    pass

x = 10
f'{x}'
