# Integers
00000
1111L
0x10000L
0X1111
0b1111L
0o12345
# strings
u"lala"
print # the grammar used should recognize this as builtin
a = [1,
     2, # after comment
     ]
def foo(a): return 1 if a == 0 else 2
class bar(object): pass
def foo2(x,y,z,): #trailing comma allowed
    pass
def bar(*baz):    #star syntax
    foo(3,4,5)
def bar2(**baz):   #star syntax
    yield; # semicolon is optional
items = []
(item for item in items)
[item for item in items]
