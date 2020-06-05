class A:
    field = 42

class Map:
    def __getitem__(self, key):
        return 42

def arguments():
    '{:{}} {:{}}'.format('one', 'two', 'three') # Noncompliant
    '{} {} {}'.format('one') # Noncompliant {{Provide a value for field(s) with index 1, 2.}}
   #^^^^^^^^^^
    '{0} {1}'.format('one') # Noncompliant {{Provide a value for field(s) with index 1.}}
    '{a} {b} {}'.format(a=1, b=2) # Noncompliant
    '{a} {b} {1}'.format(a=1, b=2) # Noncompliant {{Provide a value for field(s) with index 1.}}
    '{a} {b}'.format(a=1) # Noncompliant {{Provide a value for field "b".}}
    '{1} {a} {0}'.format(1, a=1) # Noncompliant
    '{0} {a}'.format('pos', a='one')  # Ok
    '{} {a}'.format('pos', a='one')  # Ok
    '{a} {b}'.format(a=1, b=2)  # Ok
    '{a} {a}'.format(a=1)  # Ok
    '{a} {{z}}'.format(a=1)  # Ok

    '{a}'.format(a=1, b=2)  # Out of scope (S3457)
    '{}'.format(1, 2)  # Out of scope (S3457)

def format_syntax():
    "{0".format(1)  # Noncompliant {{Fix this formatted string's syntax.}}
   #^^^^
    "0}".format(1)  # Noncompliant
    "{[".format() # Noncompliant
    "{a[]]}".format(a=0) # Noncompliant
    "}0".format() # Noncompliant
    "{a.}".format(a=A())  # Noncompliant
    "{a.field.}".format(a=A())  # Noncompliant
    "{m[attr}".format(m=Map())  # Noncompliant
    "{m[]}".format(m=Map())  # Noncompliant
    "{0}".format(1)
    "{a.field.real}".format(a=A())
    "{m[key]}".format(m=Map())
    "{m[0]}".format(m=[1])
    "{'0'}".format(**{"\'0\'": 1}) # Ok - all characters are allowed as long as they do not interfere with the format syntax
    "{m]}".format(**{"m]": 42})
    '{0!s}'.format('str')  # Ok
    '{0!z}'.format('str')  # Noncompliant  {{Fix this formatted string's syntax; !z is not a valid conversion flag.}}
    '{0} {}'.format('one', 'two') # Noncompliant {{Use only manual or only automatic field numbering, don't mix them.}}
    '{{}} {}'.format('one')  # Ok
    '{0!s:{1}}'.format('one', 'two') # Ok
    '{foo!s:{bar}}'.format(foo='one', bar='two') # Ok
    '{foo!sar}'.format(foo='a') # Noncompliant {{Fix this formatted string's syntax.}}

    # Currently we produce false-positives with some weird keys containing braces and brackets.
    '{a:{bbb]}'.format() # Noncompliant {{Fix this formatted string's syntax.}}

def nested_format():
    '{:{}} {:{}}'.format('one', 'two', 'three') # Noncompliant
    '{:{}} {:{}}'.format('one', 'two', 'three', 'four') # Ok
    '{foo:{}} {:{bar}}'.format('one', 'two', foo='three', bar='four') # Ok
    '{foo:{}} {:{bar}}'.format('one', 'two', foo='three') # Noncompliant
    '{a:{b}{c}{d}}{:{}{e}}'.format('one', a='a', b='b', c='c', d='d', e='e') # Noncompliant {{Provide a value for field(s) with index 1.}}
    '{a:{b}{c}{d}}{:{}{e}}'.format('one', 'two', a='a', b='b', c='c', d='d') # Noncompliant {{Provide a value for field "e".}}
    '{a:{0}{1}{b}}{1:{2}{2}}'.format('one', 'two', a='a', b='b') # Noncompliant {{Provide a value for field(s) with index 2.}}
    '{a:{b}{c}{d}}{:{}{e}}'.format('one', 'two', a='a', b='b', c='c', d='d', e='e') # OK
    '{a:{0}{1}{b}}{0:{2}{2}}'.format('one', 'two', 'three', a='a', b='b') # OK

def other():
    f1 = '{} {} {} {}'
#        ^^^^^^^^^^^^^>
    f1.format("1", 2, 3)  # Noncompliant
#   ^^
    f2 = '{m[0'
    f2.format(m=[1]) # Noncompliant
    f3 = '{} {}'
    f3 = '{} {} {}'
    f3.format("1") # FN

    ("{}" " {}".format(1))  # Noncompliant
    dict_data = {'one': '1', 'two': 2}
    tuple_data = ("1", 2, 3)
    '{} {} {} {}'.format(*tuple_data)  # FN
    '{one} {two} {three}'.format(**dict_data)  # FN

def fun():
    pass

def edge_case():
    var = '{} {}'.format
    var(1)  # FN
    fun = 'hello'
    fun('')
    A().format('')
    A = 1
    A.format(1.3)

