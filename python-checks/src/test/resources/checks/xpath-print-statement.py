def fib(n):
    a, b = 0, 1
    while a < n:
        print a, #Noncompliant [[sc=9;ec=17]] {{Avoid statements}}
        a, b = b, a+b
    print #Noncompliant [[sc=5;ec=10]] {{Avoid statements}}

fib(1000)
